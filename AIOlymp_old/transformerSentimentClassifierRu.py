from transformers import GPT2Tokenizer, GPT2LMHeadModel
import math
import pandas
import torch

tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
gpt3_large = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")


def calculate_perplexity(sentence, model, tokenizer):
    sentence_positive = 'довольна:' + sentence
    sentence_negative = 'недовольна:' + sentence
    list_sent = [sentence_positive, sentence_negative]
    ppl_values = []

    for sentence in list_sent:
        encodings = tokenizer(sentence, return_tensors='pt')
        input_ids = encodings.input_ids
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        ppl = math.exp(loss.item() * input_ids.size(1))
        ppl_values.append(ppl)

    if ppl_values[0] > ppl_values[1]:
        return 'отрицательный'
    elif ppl_values[0] < ppl_values[1]:
        return 'положительный'


while True:
    print(calculate_perplexity(input(), gpt3_large, tokenizer))
