from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss
import pandas
import torch

print('Loading data...')
data = pandas.read_csv('rucode-7.0/public_test.csv')
# data = data.iloc[:100]
print('Generating prompts...')
people_sentences = list('Человек и нейросеть должны были продолжить диалог:\n' + data['context']
                        + '\n\n\nОдин из них сделал это так:\n' + data['answer']
                        + '\n\n\nБыло ли это продолжение сгенерировано нейросетью?\nНет')
ai_sentences = list('Человек и нейросеть должны были продолжить диалог:\n' + data['context']
                    + '\n\n\nОдин из них сделал это так:\n' + data['answer']
                    + '\n\n\nБыло ли это продолжение сгенерировано нейросетью?\nДа')

print('Loading tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2", pad_token='[pad]')
print('Loading model...')
rugpt3_large = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
rugpt3_large.resize_token_embeddings(len(tokenizer))


def calculate_loss(sentences, tokenizer, model):
    batch_size = 10
    all_loss = []
    for batch_start in range(0, len(sentences), batch_size):
        print(f'Processing batch {batch_start // batch_size + 1:>2} '
              f'({batch_start:>4}/{len(sentences)} sentences processed)')
        batch = sentences[batch_start:batch_start + batch_size]
        # tokens = [[token5izer.bos_token_id] + tokenizer.encode(sentence) + [tokenizer.eos_token_id]
        #           for sentence in batch]
        # inputs = pad_sequence([torch.LongTensor(sequence) for sequence in tokens], batch_first=True, padding_value=0)
        #
        # mask = (inputs != tokenizer.pad_token_id).float()
        # labels = inputs.masked_fill(inputs == tokenizer.pad_token_id, 0)
        embeddings = tokenizer(batch, return_tensors='pt', padding='longest')

        with torch.no_grad():
            outputs = model(input_ids=embeddings.input_ids, labels=embeddings.input_ids,
                            attention_mask=embeddings.attention_mask)

        loss_fn = CrossEntropyLoss(reduction='none')
        loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), embeddings.input_ids.view(-1))
        average_loss = loss.view(outputs.logits.size(0), -1).mean(dim=1)

        all_loss.append(average_loss)
    return torch.concat(all_loss, dim=0)


print('Calculating perplexity...')
people_loss = calculate_loss(people_sentences, tokenizer, rugpt3_large)
ai_loss = calculate_loss(ai_sentences, tokenizer, rugpt3_large)
loss_delta = ai_loss - people_loss
best = (loss_delta > loss_delta.median()).int()

print('Saving result...')
with open('rucode-7.0/test_result.csv', 'w') as f:
    for which in best:
        f.write(['people', 'ai'][which] + '\n')
