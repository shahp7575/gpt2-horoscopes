import os
import torch
import config
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def make_prompt(category):
    return f"<|category|> {category} <|horoscope|>"

def generate(model, tokenizer, temperature, num_outputs, top_k):
    
#     https://huggingface.co/transformers/main_classes/model.html?highlight=generate
    sample_outputs = model.generate(generated, 
                                    #bos_token_id=random.randint(1,30000),
                                    do_sample=True,   
                                    top_k=top_k, 
                                    max_length = 300,
                                    top_p=0.95,
                                    temperature=temperature,
                                    num_return_sequences=num_outputs)
    
    return sample_outputs
    
    
if __name__ == "__main__":
    category = 'birthday'
    prompt = make_prompt(category)
    
    model = GPT2LMHeadModel.from_pretrained(config.MODEL_DIR)
    tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(config.MODEL_DIR, 'tokenizer'))
    device = torch.device(config.DEVICE)

    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)
    
    model.to(device)
    model.eval()

    sample_outputs = generate(model, tokenizer, temperature=0.95, num_outputs=3, top_k=40)
    
    for i, sample_output in enumerate(sample_outputs):
        final_out = tokenizer.decode(sample_output, skip_special_tokens=True)
        print(f"{i}: {final_out[len(category):]}\n\n")