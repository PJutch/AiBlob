from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy
import torch

dataset = load_dataset("yelp_review_full")

small_train_dataset = dataset["train"].shuffle(seed=42).select(range(100))
small_eval_dataset = dataset["test"].shuffle(seed=42).select(range(100))

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize(dataset: Dataset) -> Dataset:
    return dataset.map(
        lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True),
        batched=True
    )


small_train_dataset = tokenize(small_train_dataset)
small_eval_dataset = tokenize(small_eval_dataset)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Training on {device}")
model.to(device)

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = numpy.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model('model2')
