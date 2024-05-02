from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import torch as th

def load_model(model_name_or_path):
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=th.float16,
    )
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        ignore_mismatched_sizes=True,
        torch_dtype=th.float16,
        quantization_config=bnb_config,
        device_map="auto",
    )
    llm_model.tie_weights()
    config = llm_model.config
    config.num_labels = 1
    config.num_attention_heads = 16

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return llm_model, config, tokenizer
