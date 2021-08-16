import os
import torch
import warnings
from transformers import AutoTokenizer, AutoModelWithLMHead

warnings.filterwarnings("ignore")

def make_prompt(category):
    return f"<|category|> {category} <|horoscope|>"

def generate(prompt, model, tokenizer, temperature, num_outputs, top_k):
    
#     https://huggingface.co/transformers/main_classes/model.html?highlight=generate
    sample_outputs = model.generate(prompt, 
                                    #bos_token_id=random.randint(1,30000),
                                    do_sample=True,   
                                    top_k=top_k, 
                                    max_length = 300,
                                    top_p=0.95,
                                    temperature=temperature,
                                    num_return_sequences=num_outputs)
    
    return sample_outputs
    
    
if __name__ == "__main__":
    category = 'career'
    prompt = make_prompt(category)
    
    tokenizer = AutoTokenizer.from_pretrained('shahp7575/gpt2-horoscopes')
    model = AutoModelWithLMHead.from_pretrained('shahp7575/gpt2-horoscopes')

    prompt_encoded = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

    sample_output = generate(prompt_encoded, model, tokenizer, temperature=0.95, num_outputs=1, top_k=40)
    final_out = tokenizer.decode(sample_output[0], skip_special_tokens=True)
    print(final_out[len(category)+2:])