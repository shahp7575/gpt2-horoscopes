## GPT2-Horoscopes

GPT2 model fine-tuned on horoscopes data from https://www.horoscope.com/us/index.aspx. Rather than just fine-tuning on the horscope texts, it also adds additional metadata i.e. horoscope category. There are 5 categories - *general, love, career, wellness, birthday*. The model learns to associate each category with its type of texts.

The final fine-tuned model takes as input the *category* and generates horoscopes.