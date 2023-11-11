from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
import evaluate
import numpy

imdb = load_dataset('imdb')

small_train_dataset = imdb["train"].shuffle().select(range(100))
small_eval_dataset = imdb["test"].shuffle().select(range(100))

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
tokenized_train = small_train_dataset.map(lambda data: tokenizer(data['text'], truncation=True))
tokenized_eval = small_eval_dataset.map(lambda data: tokenizer(data['text'], truncation=True))

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

metric_ = evaluate.load('accuracy')


def metric(eval_res):
    logits, labels = eval_res
    pred = numpy.argmax(logits, axis=1)
    return metric_.compute(predictions=pred, references=labels)


id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    # push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=metric,
)
trainer.train()
trainer.save_model('en_classifier2')
