import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def grab_model_info():
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    input_text = "Hello, I am a"
    input_ids = tokenizer(input_text, return_tensors="pt")
    return tokenizer, input_ids, model 


def generate_outputs(input_ids, tokenizer, model):
    output = model.generate(input_ids.input_ids, max_length=50)
    print(tokenizer.decode(output[0]))
