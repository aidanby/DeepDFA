import sys
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from peft import PeftModel


def load_model_tokenizer(
    path,
    model_type=None,
    peft_path=None,
    torch_dtype=torch.float16,
    quantization=None,
    eos_token=None,
    pad_token=None,
):
    """
    load model and tokenizer by transfromers
    """

    # load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tokenizer.padding_side = "left"

    config, unused_kwargs = AutoConfig.from_pretrained(
        path, trust_remote_code=True, return_unused_kwargs=True
    )
    print("unused_kwargs:", unused_kwargs)
    print("config input:\n", config)

    if eos_token:
        eos_token = eos_token
        eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
        print(f"eos_token {eos_token} from user input")
    else:
        if hasattr(config, "eos_token_id") and config.eos_token_id:
            print(f"eos_token_id {config.eos_token_id} from config.json")
            eos_token_id = config.eos_token_id
            eos_token = tokenizer.convert_ids_to_tokens(config.eos_token_id)
        elif hasattr(config, "eos_token") and config.eos_token:
            print(f"eos_token {config.eos_token} from config.json")
            eos_token = config.eos_token
            eos_token_id = tokenizer.convert_tokens_to_ids(config.eos_token)
        else:
            raise ValueError(
                "No available eos_token or eos_token_id, please provide eos_token by params or eos_token_id by config.json"
            )

    if pad_token:
        pad_token = pad_token
        pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
        print(f"pad_token {pad_token} from user input")
    else:
        if hasattr(config, "pad_token_id") and config.pad_token_id:
            print(f"pad_token_id {config.pad_token_id} from config.json")
            pad_token_id = config.pad_token_id
            pad_token = tokenizer.convert_ids_to_tokens(config.pad_token_id)
        elif hasattr(config, "pad_token") and config.pad_token:
            print(f"pad_token {config.pad_token} from config.json")
            pad_token = config.pad_token
            pad_token_id = tokenizer.convert_tokens_to_ids(config.pad_token)
        else:
            print(f"pad_token {eos_token} duplicated from eos_token")
            pad_token = eos_token
            pad_token_id = eos_token_id

    # update tokenizer eos_token and pad_token
    tokenizer.eos_token_id = eos_token_id
    tokenizer.eos_token = eos_token
    tokenizer.pad_token_id = pad_token_id
    tokenizer.pad_token = pad_token

    print(
        f"tokenizer's eos_token: {tokenizer.eos_token}, pad_token: {tokenizer.pad_token}"
    )
    print(
        f"tokenizer's eos_token_id: {tokenizer.eos_token_id}, pad_token_id: {tokenizer.pad_token_id}"
    )
    print(tokenizer)

    bnb_config = BitsAndBytesConfig(
        bnb_4bit_use_double_quant=True,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        path,
        config=config,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    if peft_path:
        print("Loading PEFT MODEL...")
        model = PeftModel.from_pretrained(base_model, peft_path)
    else:
        print("Loading Original MODEL...")
        model = base_model

    model.eval()

    print(
        "=======================================MODEL Configs====================================="
    )
    print(model.config)
    print(
        "========================================================================================="
    )
    print(
        "=======================================MODEL Archetecture================================"
    )
    print(model)
    print(
        "========================================================================================="
    )

    return model, tokenizer


def hf_inference(
    model, tokenizer, text_list, args=None, max_new_tokens=512, do_sample=True, **kwargs
):
    """
    transformers models inference by huggingface
    """
    inputs = tokenizer(
        text_list, return_tensors="pt", padding=True, add_special_tokens=False
    ).to("cuda")
    print(
        "================================Prompts and Generations============================="
    )

    outputs = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **kwargs,
    )

    gen_text = tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    for i in range(len(text_list)):
        print("=========" * 10)
        print(f"Prompt:\n{text_list[i]}")
        gen_text[i] = gen_text[i].replace(tokenizer.pad_token, "")
        print(f"Generation:\n{gen_text[i]}")
        sys.stdout.flush()

    return gen_text


class PeftInference:
    def __init__(self, base_model, lora_adapter):
        self.base_model = base_model
        self.lora_adapter = lora_adapter

    def load_model(self):
        model, tokenizer = load_model_tokenizer(
            self.base_model,
            model_type="",
            peft_path=self.lora_adapter,
            eos_token="</s>",
            pad_token="<unk>",
            quantization="4bit",
        )
        return model, tokenizer
