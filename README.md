# GPT2-Horoscopes
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/shahp7575/gpt2-horoscopes-app/generate.py)

## Model Description
GPT2 fine-tuned on Horoscopes dataset scraped from [Horoscopes.com](https://www.horoscope.com/us/index.aspx). This model generates horoscopes given a horoscope *category*.

Try it out on the Streamlit [web app](https://share.streamlit.io/shahp7575/gpt2-horoscopes-app/generate.py)!

## Uses & Limitations

### How to use
The model can be used directly with the HuggingFace `pipeline` API.
```python
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("shahp7575/gpt2-horoscopes")
model = AutoModelWithLMHead.from_pretrained("shahp7575/gpt2-horoscopes")
```

### Generation

Input Text Format - `<|category|> {category_type} <|horoscope|>`

Supported Categories - *general, career, love, wellness, birthday*

Example:
```python
prompt = <|category|> career <|horoscope|>
prompt_encoded = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
sample_outputs = model.generate(prompt, 
                                do_sample=True,   
                                top_k=40, 
                                max_length = 300,
                                top_p=0.95,
                                temperature=0.95,
                                num_return_sequences=1)
```

### Training Data
Dataset is scraped from [Horoscopes.com](https://www.horoscope.com/us/index.aspx) for 5 categories with a total of ~12k horoscopes. The dataset will be made public soon.

### Training Procedure
The model uses the [GPT2](https://huggingface.co/gpt2) checkpoint and then is fine-tuned on horoscopes dataset for 5 different categories. Since the goal of the fine-tuned model was also to understand different horoscopes for different category types, the *categories* are added to the training data separated by special token `<|category|>`. 

**Training Parameters:**

- EPOCHS = 5
- LEARNING RATE = 5e-4
- WARMUP STEPS = 1e2
- EPSILON = 1e-8
- SEQUENCE LENGTH = 300

### Evaluation Results

Loss: 2.77

### Limitations
This model is only fine-tuned on horoscopes by categories. They do not, and neither attempt to, represent actual horoscopes. It is developed only for educational and learning purposes.

## References

- [Rey Farhan's - Fine-tuning GPT2 Notebook](https://colab.research.google.com/drive/13dZVYEOMhXhkXWfvSMVM1TTtUDrT6Aeh?usp=sharing#scrollTo=_U3m6wr3Ahzt)

- [Jonathan Bgn - Building a Slogan Generator with GPT-2](https://jonathanbgn.com/gpt2/2020/01/20/slogan-generator.html)