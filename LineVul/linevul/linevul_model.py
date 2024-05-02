import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    RobertaForSequenceClassification,
    AutoModelForSequenceClassification,
    LlamaForSequenceClassification,
)


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, extra_dim):
        super().__init__()
        # TODO: add dim on just dense or both?
        self.dense = nn.Linear(config.hidden_size + extra_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, flowgnn_embed, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        if flowgnn_embed is not None:
            x = torch.cat((x, flowgnn_embed), dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


# Llama model classification. Automodel does not currently work for custom model inheritance.
class LLMModel(LlamaForSequenceClassification):
    def __init__(self, encoder, flowgnn_encoder, config, tokenizer, args):
        super(LLMModel, self).__init__(config=config)
        self.encoder = encoder
        if not args.no_flowgnn:
            self.flowgnn_encoder = flowgnn_encoder
        self.tokenizer = tokenizer
        self.args = args

    def forward(
        self,
        input_ids=None,
        input_embed=None,
        output_attentions=False,
    ):
        if output_attentions:
            if input_ids is not None:
                input_ids = input_ids[:, -1, :]
                self.encoder.eval()
                outputs = self.encoder(
                    input_ids,
                    attention_mask=input_ids.ne(1),
                    output_attentions=output_attentions,
                )
            else:
                outputs = self.encoder(
                    inputs_embeds=input_embed, output_attentions=output_attentions
                )
            attentions = outputs.attentions
            last_hidden_state = outputs.last_hidden_state
            return last_hidden_state, attentions
        else:
            if input_ids is not None:
                input_ids = input_ids[:, -1, :]
                input_ids = input_ids.to(self.encoder.device)
                attention_mask = input_ids.ne(1)
                attention_mask.to(self.encoder.device)
                print(f"input_ids device: {input_ids.device}")
                print(f"encoder device: {self.encoder.device}")
                outputs = self.encoder(
                    input_ids,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )[0]
            else:
                outputs = self.encoder(
                    inputs_embeds=input_embed, output_attentions=output_attentions
                )[0]
            return outputs, None


class GNNModel(LlamaForSequenceClassification):
    def __init__(self, flowgnn_encoder, config, args):
        super(GNNModel, self).__init__(config=config)
        if not args.no_flowgnn:
            self.flowgnn_encoder = flowgnn_encoder
        self.classifier = ClassificationHead(
            config, 0 if args.no_flowgnn else self.flowgnn_encoder.out_dim
        )
        self.args = args

    def forward(
        self,
        labels=None,
        graphs=None,
        llm_hidden_states=None,
        llm_attentions=None,
        output_attentions=False,
    ):
        if output_attentions and llm_attentions is not None:
            print(f"llm_hidden_states shape: {llm_hidden_states.shape}")
            print(f"flowgnn_embed shape: {flowgnn_embed.shape}")

            # reshape llm_hidden_states to match the shape of flowgnn
            llm_hidden_states = llm_hidden_states.unsqueeze(1).repeat(
                1, graphs[0].num_nodes, 1
            )
            print(f"llm_hidden_states shape: {llm_hidden_states.shape}")
            print(f"flowgnn_embed shape: {flowgnn_embed.shape}")

            logits = self.classifier(llm_hidden_states, flowgnn_embed)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob, llm_attentions
            else:
                return prob, llm_attentions
        else:
            if self.args.no_flowgnn:
                flowgnn_embed = None
            elif graphs is not None:
                flowgnn_embed = self.flowgnn_encoder(graphs, {})
            logits = self.classifier(llm_hidden_states, flowgnn_embed)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob
            else:
                return prob
