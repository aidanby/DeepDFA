# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function
import argparse
import logging
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
import random
import re
import json
import numpy as np
import torch
import sys
import os.path as path
from tqdm import tqdm
from linevul_model import LLMModel, GNNModel
import pandas as pd
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)


# metrics
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# word-level tokenizer
from tokenizers import Tokenizer

deepdfa = path.abspath(path.join(__file__, "../../../DDFA/"))
sys.path.append(deepdfa)
from code_gnn.models.flow_gnn.ggnn import FlowGNNGGNNModule
from sastvd.linevd.datamodule import BigVulDatasetLineVDDataModule
from code_gnn.models.hf_inference import PeftInference
import dgl
from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self, input_tokens, input_ids, label, i):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label
        self.index = i


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_type="train", return_index=False):
        if file_type == "train":
            file_path = args.train_data_file
        elif file_type == "eval":
            file_path = args.eval_data_file
        elif file_type == "test":
            file_path = args.test_data_file
        self.examples = []

        if file_path.endswith(".jsonl"):
            df = pd.read_json(file_path, orient="records", lines=True)
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path, index_col=0)
        else:
            raise NotImplementedError(file_path)
        if "holdout" in os.path.basename(file_path):
            df["split"].replace("holdout", "test")

        # if args.sample:
        df = df.sample(100)

        if "processed_func" in df.columns:
            func_key = "processed_func"
        elif "func" in df.columns:
            func_key = "func"
        if args.train_data_file is not None and "devign" in args.train_data_file:

            def zonk(s):
                lines = s.splitlines()
                lines = [
                    re.sub(r"[\t ]+", " ", l.strip())
                    for l in lines
                    if len(l.strip()) > 0
                ]
                return "\n".join(lines)

            df[func_key] = df[func_key].apply(zonk)
        funcs = df[func_key].tolist()
        labels = df["target"].astype(int).tolist()
        indices = df.index.astype(int).tolist()
        for i in tqdm(range(len(funcs)), desc="load dataset"):
            self.examples.append(
                convert_examples_to_features(
                    funcs[i], labels[i], tokenizer, args, indices[i]
                )
            )
        if file_type in ("train", "test"):
            for example in self.examples[:3]:
                logger.info("*** Example ***")
                logger.info("label: {}".format(example.label))
                logger.info(
                    "input_tokens: {}".format(
                        [x.replace("\u0120", "_") for x in example.input_tokens]
                    )
                )
                logger.info(
                    "input_ids: {}".format(" ".join(map(str, example.input_ids)))
                )
        self.return_index = args.eval_export or return_index

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        if self.return_index:
            return (
                torch.tensor(self.examples[i].input_ids),
                torch.tensor(self.examples[i].label),
                torch.tensor(self.examples[i].index),
            )
        else:
            return torch.tensor(self.examples[i].input_ids), torch.tensor(
                self.examples[i].label
            )


def convert_examples_to_features(func, label, tokenizer, args, i):
    if args.use_word_level_tokenizer:
        encoded = tokenizer.encode(func)
        encoded = encoded.ids
        if len(encoded) > 510:
            encoded = encoded[:510]
        encoded.insert(0, 0)
        encoded.append(2)
        if len(encoded) < 512:
            padding = 512 - len(encoded)
            for _ in range(padding):
                encoded.append(1)
        source_ids = encoded
        source_tokens = []
        return InputFeatures(source_tokens, source_ids, label, i)

    # Llama tokenizer adds cls and eos tokens automatically, but does not have pad token by default
    # Setting pad token as eos token similarly to GPT-2 training
    tokenizer.pad_token = tokenizer.eos_token
    source_tokens = tokenizer.tokenize(str(func))
    source_tokens = source_tokens[: args.block_size]
    source_ids = tokenizer(
        str(func),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.block_size,
    )["input_ids"].to(args.device)
    return InputFeatures(source_tokens, source_ids, label, i)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    dgl.seed(args.seed)


def train(
    args,
    train_dataset,
    hf_model,
    gnn_model,
    tokenizer,
    eval_dataset,
    flowgnn_dataset,
):
    """Train the model"""
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=0,
    )

    args.max_steps = args.epochs * len(train_dataloader)
    # evaluate the model per 10 epochs
    args.save_steps = len(train_dataloader) * 10
    args.warmup_steps = args.max_steps // 5

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in gnn_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in gnn_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
    )
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d",
        args.train_batch_size // max(args.n_gpu, 1),
    )
    logger.info(
        "  Total train batch size = %d",
        args.train_batch_size * args.gradient_accumulation_steps,
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1 = 0
    gnn_model.zero_grad()
    num_missing = 0
    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (inputs_ids, labels, index) = [x.to(args.device) for x in batch]

            if flowgnn_dataset is None:
                graphs = None
            else:
                graphs, keep_idx = flowgnn_dataset.get_indices(index)
                num_missing += len(labels) - len(keep_idx)
                inputs_ids = inputs_ids[keep_idx]
                labels = labels[keep_idx]
                graphs.to(args.device)
            gnn_model.train()

            # Receive LLM final hidden states and send to device (default cuda:0)
            llm_hidden_states = hf_model(input_ids=inputs_ids)
            # GNN model forward pass
            loss, logits = gnn_model(
                labels=labels,
                graphs=graphs,
                llm_hidden_states=llm_hidden_states,
            )

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss

            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(
                    np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4
                )

                if global_step % args.save_steps == 0:
                    results = evaluate(
                        args,
                        hf_model,
                        gnn_model,
                        tokenizer,
                        eval_dataset,
                        flowgnn_dataset,
                        eval_when_training=True,
                    )

                    # Save model checkpoint
                    if results["eval_f1"] > best_f1:
                        best_f1 = results["eval_f1"]
                        logger.info("  " + "*" * 20)
                        logger.info("  Best f1:%s", round(best_f1, 4))
                        logger.info("  " + "*" * 20)

                        checkpoint_prefix = "checkpoint-best-f1"
                        output_dir = os.path.join(
                            args.output_dir, "{}".format(checkpoint_prefix)
                        )
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            gnn_model.module
                            if hasattr(gnn_model, "module")
                            else gnn_model
                        )
                        output_dir = os.path.join(
                            output_dir, "{}".format(args.model_name)
                        )
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
        logger.info("%d items missing", num_missing)
        checkpoint_prefix = "checkpoint-last"
        output_dir = os.path.join(args.output_dir, "{}".format(checkpoint_prefix))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = gnn_model.module if hasattr(gnn_model, "module") else gnn_model
        output_dir = os.path.join(output_dir, "{}".format(args.model_name))
        torch.save(model_to_save.state_dict(), output_dir)
        logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(
    args, hf_model, gnn_model, tokenizer, eval_dataset, flowgnn_dataset, eval_when_training=False
):
    # build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        num_workers=0,
    )

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    num_missing = 0
    eval_loss = 0.0
    nb_eval_steps = 0
    gnn_model.eval()
    logits = []
    y_trues = []
    for batch in tqdm(eval_dataloader, desc="evaluate eval"):
        (inputs_ids, labels, index) = [x.to(args.device) for x in batch]

        if flowgnn_dataset is None:
            graphs = None
        else:
            graphs, keep_idx = flowgnn_dataset.get_indices(index)
            num_missing += len(labels) - len(keep_idx)
            inputs_ids = inputs_ids[keep_idx]
            labels = labels[keep_idx]
        with torch.no_grad():
            llm_hidden_states = hf_model(input_ids=inputs_ids)
            lm_loss, logit = gnn_model(
                llm_hidden_states=llm_hidden_states, labels=labels, graphs=graphs
            )
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    logger.info("%d items missing", num_missing)

    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5
    best_f1 = 0
    y_preds = logits[:, 1] > best_threshold
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test(
    args,
    hf_model,
    gnn_model,
    tokenizer,
    test_dataset,
    flowgnn_dataset,
    output_dir,
    best_threshold=0.5,
):
    # build dataloader
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=args.eval_batch_size,
        num_workers=0,
    )

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    num_missing = 0
    eval_loss = 0.0
    nb_eval_steps = 0
    gnn_model.eval()
    logits = []
    y_trues = []
    if args.profile:
        prof = FlopsProfiler(gnn_model)
    if args.time:
        pass
    profs = []
    for i, batch in enumerate(tqdm(test_dataloader, desc="evaluate test")):
        do_profile = args.profile and i > 2
        do_time = args.time and i > 2
        if do_profile:
            prof.start_profile()
        if do_time:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        (inputs_ids, labels, index) = [x.to(args.device) for x in batch]

        if flowgnn_dataset is None:
            graphs = None
        else:
            graphs, keep_idx = flowgnn_dataset.get_indices(index)
            num_missing += len(labels) - len(keep_idx)
            inputs_ids = inputs_ids[keep_idx]
            labels = labels[keep_idx]
        with torch.no_grad():
            if do_time:
                start.record()
            llm_hidden_states = hf_model(input_ids=inputs_ids)
            lm_loss, logit = gnn_model(
                llm_hidden_states=llm_hidden_states, labels=labels, graphs=graphs
            )
            if do_time:
                end.record()
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
        if do_profile:
            flops = prof.get_total_flops(as_string=True)
            params = prof.get_total_params(as_string=True)
            macs = prof.get_total_macs(as_string=True)
            prof.print_model_profile(
                profile_step=i, output_file=f"{output_dir}.profile.txt"
            )
            prof.end_profile()
            logger.info("step %d: %s flops %s params %s macs", i, flops, params, macs)
            profs.append(
                {
                    "step": i,
                    "flops": flops,
                    "params": params,
                    "macs": macs,
                    "batch_size": len(labels),
                }
            )
        if do_time:
            torch.cuda.synchronize()
            tim = start.elapsed_time(end)
            logger.info("step %d: time %f", i, tim)
            profs.append(
                {
                    "step": i,
                    "batch_size": len(labels),
                    "runtime": tim,
                }
            )
    if args.profile:
        filename = f"{output_dir}.profiledata.txt"
    elif args.time:
        filename = f"{output_dir}.timedata.txt"
    else:
        filename = None
    if filename is not None:
        with open(filename, "w") as f:
            json.dump(profs, f)
    logger.info("%d items missing", num_missing)

    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = logits[:, 1] > best_threshold
    print("classification_report")
    print(classification_report(y_trues, y_preds))
    print("confusion_matrix")
    print(confusion_matrix(y_trues, y_preds))
    acc = accuracy_score(y_trues, y_preds)
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)
    result = {
        "test_accuracy": float(acc),
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1),
        "test_threshold": best_threshold,
    }

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))


def main():
    parser = argparse.ArgumentParser()
    ## parameters
    parser.add_argument(
        "--train_data_file",
        default=None,
        type=str,
        required=False,
        help="The input training data file (a csv file).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type",
        default="bert",
        type=str,
        help="The model architecture to be fine-tuned.",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--test_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--model_name", default="model.bin", type=str, help="Saved model name."
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--use_non_pretrained_model",
        action="store_true",
        default=False,
        help="Whether to use non-pretrained model.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--code_length",
        default=256,
        type=int,
        help="Optional Code input sequence length after tokenization.",
    )

    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--eval_export", action="store_true", help="Whether to save prediction output."
    )
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")

    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_local_explanation",
        default=False,
        action="store_true",
        help="Whether to do local explanation. ",
    )
    parser.add_argument(
        "--reasoning_method",
        default=None,
        type=str,
        help="Should be one of 'attention', 'shap', 'lime', 'lig'",
    )

    parser.add_argument(
        "--train_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument("--epochs", type=int, default=1, help="training epochs")
    # RQ2
    parser.add_argument(
        "--effort_at_top_k",
        default=0.2,
        type=float,
        help="Effort@TopK%Recall: effort at catching top k percent of vulnerable lines",
    )
    parser.add_argument(
        "--top_k_recall_by_lines",
        default=0.01,
        type=float,
        help="Recall@TopK percent, sorted by line scores",
    )
    parser.add_argument(
        "--top_k_recall_by_pred_prob",
        default=0.2,
        type=float,
        help="Recall@TopK percent, sorted by prediction probabilities",
    )

    parser.add_argument(
        "--do_sorting_by_line_scores",
        default=False,
        action="store_true",
        help="Whether to do sorting by line scores.",
    )
    parser.add_argument(
        "--do_sorting_by_pred_prob",
        default=False,
        action="store_true",
        help="Whether to do sorting by prediction probabilities.",
    )
    # RQ3 - line-level evaluation
    parser.add_argument(
        "--top_k_constant", type=int, default=10, help="Top-K Accuracy constant"
    )
    # num of attention heads
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=16,
        help="number of attention heads",
    )
    # raw predictions
    parser.add_argument(
        "--write_raw_preds",
        default=False,
        action="store_true",
        help="Whether to write raw predictions on test data.",
    )
    # word-level tokenizer
    parser.add_argument(
        "--use_word_level_tokenizer",
        default=False,
        action="store_true",
        help="Whether to use word-level tokenizer.",
    )
    # bpe non-pretrained tokenizer
    parser.add_argument(
        "--use_non_pretrained_tokenizer",
        default=False,
        action="store_true",
        help="Whether to use non-pretrained bpe tokenizer.",
    )
    # finetuned model
    parser.add_argument(
        "--use_finetuned_model",
        action="store_true",
        default=False,
        help="Whether to use finetuned LLM model",
    )
    parser.add_argument(
        "--finetuned_path",
        default="/home/checkpoints_codellama7/step_1200",
        type=str,
    )

    parser.add_argument(
        "--no_flowgnn",
        action="store_true",
        help="do not train/evaluate DDFA as part of the model",
    )
    parser.add_argument(
        "--really_no_flowgnn", action="store_true", help="do not load any DDFA stuff"
    )
    parser.add_argument(
        "--no_concat",
        action="store_true",
        help="do not concatenate DDFA abstract dataflow embedding",
    )
    parser.add_argument(
        "--dsname", type=str, default="bigvul", help="dataset name to load for DDFA"
    )
    parser.add_argument("--profile", action="store_true", help="profile MACs")
    parser.add_argument("--time", action="store_true", help="measure inference time")
    args = parser.parse_args()

    # Setup CUDA, GPU
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "device: %s, n_gpu: %s",
        device,
        args.n_gpu,
    )

    # Set seed
    set_seed(args)

    # Load LLM
    if args.use_word_level_tokenizer:
        print("using wordlevel tokenizer!")
        tokenizer = Tokenizer.from_file("./word_level_tokenizer/wordlevel.json")
    elif args.use_non_pretrained_tokenizer:
        tokenizer = AutoTokenizer(
            vocab_file="bpe_tokenizer/bpe_tokenizer-vocab.json",
            merges_file="bpe_tokenizer/bpe_tokenizer-merges.txt",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.padding_side = "left"

    if args.use_finetuned_model:
        logger.info(f"Loading finetuned model from {args.model_name_or_path} ")
        peft_inference = PeftInference(
            args.model_name_or_path,
            args.finetuned_path,
        )
        llm_model, tokenizer = peft_inference.load_model()

    else:
        logger.info(f"Loading pretrained model from {args.model_name_or_path} ")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        llm_model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            output_hidden_states=True,
            device_map="balanced",
        )

    # Set manual config options
    llm_model.config.num_labels = 1
    llm_model.config.num_attention_heads = args.num_attention_heads
    llm_model.config.pad_token_id = llm_model.model.config.eos_token_id

    # load all graphs
    if args.really_no_flowgnn:
        flowgnn_datamodule = None
        flowgnn_dataset = None
    else:
        feat = "_ABS_DATAFLOW_datatype_all_limitall_1000_limitsubkeys_1000"
        gtype = "cfg"
        label_style = "graph"
        dsname = args.dsname
        concat_all_absdf = not args.no_concat
        flowgnn_datamodule = BigVulDatasetLineVDDataModule(
            feat,
            gtype,
            label_style,
            dsname,
            undersample=None,
            oversample=None,
            sample=-1,
            sample_mode=args.sample,
            train_workers=1,
            val_workers=0,
            test_workers=0,
            split="fixed",
            batch_size=256,
            seed=args.seed,
            concat_all_absdf=concat_all_absdf,
            # use_weighted_loss=False,
            # use_random_weighted_sampler=False,
            train_includes_all=True,
            load_features=not args.no_flowgnn,
        )
        flowgnn_dataset = flowgnn_datamodule.train
        logger.info("FlowGNN dataset:\n%s", flowgnn_datamodule.train.df)

    # Load model
    if args.really_no_flowgnn:
        flowgnn_model = None
    else:
        input_dim = flowgnn_datamodule.input_dim
        hidden_dim = 32
        n_steps = 5
        num_output_layers = 3
        flowgnn_model = FlowGNNGGNNModule(
            feat,
            input_dim,
            hidden_dim,
            n_steps,
            num_output_layers,
            label_style=label_style,
            concat_all_absdf=concat_all_absdf,
            # undersample_node_on_loss_factor=None,
            # test_every=False,
            # tune_nni=False,
            # positive_weight=None,
            encoder_mode=True,
        )
        logger.info("FlowGNN output dim: %d", flowgnn_model.out_dim)

    # Combined model
    # if flowgnn_model:
    #     flowgnn_model = torch.nn.DataParallel(flowgnn_model)
    #     flowgnn_model = flowgnn_model.module

    llm_model = LLMModel(llm_model, flowgnn_model, llm_model.config, tokenizer, args)
    gnn_model = GNNModel(flowgnn_model, llm_model.config, args)

    # print number of params
    def count_params(model):
        if model is None:
            return 0
        return sum(p.numel() for p in model.parameters())

    params = count_params(llm_model.encoder) + count_params(gnn_model.classifier)
    if not args.no_flowgnn:
        params += count_params(llm_model.flowgnn_encoder)
    print("parameters:", params)
    print("encoder:", llm_model.encoder)
    print("classifier:", gnn_model.classifier)
    if not args.no_flowgnn:
        print("flowgnn_encoder:", llm_model.flowgnn_encoder)

    gnn_model.to(args.device)
    # if args.n_gpu > 1:
    #     gnn_model = torch.nn.DataParallel(gnn_model)
    # Training
    if args.do_train:
        train_dataset = TextDataset(
            tokenizer, args, file_type="train", return_index=True
        )
        eval_dataset = TextDataset(tokenizer, args, file_type="eval", return_index=True)
        train(
            args,
            train_dataset,
            llm_model,
            gnn_model,
            tokenizer,
            eval_dataset,
            flowgnn_dataset,
        )
    # Evaluation
    if args.do_eval:
        output_dir = os.path.join(
            args.output_dir, args.model_name, "checkpoint-best-f1", "model.bin"
        )
        gnn_model.load_state_dict(torch.load(output_dir))
        evaluate(args, llm_model, gnn_model, tokenizer)
    # Test
    if args.do_test:
        if os.path.exists(args.model_name):
            output_dir = args.model_name
        else:
            checkpoint_prefix = f"checkpoint-best-f1/{args.model_name}"
            output_dir = os.path.join(args.output_dir, checkpoint_prefix)
            if not os.path.exists(output_dir):
                output_dir = os.path.join(
                    args.output_dir, "checkpoint-best-f1", args.model_name
                )
            if not os.path.exists(output_dir):
                output_dir = os.path.join(
                    args.output_dir, args.model_name, "checkpoint-best-f1", "model.bin"
                )
            if not os.path.exists(output_dir):
                # linevul/saved_models/dataset_size/imbalanced_0.01/checkpoint-best-f1
                output_dir = os.path.join(
                    args.output_dir, args.model_name, "checkpoint-best-f1", "model.bin"
                )
        gnn_model.load_state_dict(torch.load(output_dir, map_location=args.device))
        test_dataset = TextDataset(tokenizer, args, file_type="test", return_index=True)
        test(
            args,
            llm_model,
            gnn_model,
            tokenizer,
            test_dataset,
            flowgnn_dataset,
            output_dir,
            best_threshold=0.5,
        )


if __name__ == "__main__":
    main()
