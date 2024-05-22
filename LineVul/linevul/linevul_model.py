import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    RobertaForSequenceClassification,
    LlamaForSequenceClassification,
    LlamaForCausalLM,
)
import gc  # garbage collect library


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, extra_dim):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size + extra_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_dropout)
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
    ):
        if input_ids is not None:
            input_ids = input_ids[:, -1, :]
            self.encoder.eval()
            attention_mask = input_ids.ne(1)
            outputs = self.encoder(
                input_ids, attention_mask=attention_mask, output_hidden_states=True
            )
        else:
            outputs = self.encoder(inputs_embeds=input_embed, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        attention_hidden_states = hidden_states[1:]
        final_attention_states = attention_hidden_states[-1]
        return final_attention_states


class GNNModel(nn.Module):
    def __init__(self, flowgnn_encoder, config, args):
        super().__init__()
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
    ):
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
