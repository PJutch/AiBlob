from datasets import load_dataset
from transformers import (AutoImageProcessor, DefaultDataCollator, AutoModelForImageClassification,
                          TrainingArguments, Trainer)
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
import evaluate
import numpy

food = load_dataset("food101", split="train[:5000]")
food = food.train_test_split(test_size=0.2)
# food['train'][0]['image'].show()

labels = food["train"].features["label"].names
label2id = {label: str(i) for i, label in enumerate(labels)}
id2label = {str(i): label for i, label in enumerate(labels)}

checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

transforms = Compose([
    RandomResizedCrop(image_processor.size['shortest_edge']
                      if 'shortest_edge' in image_processor.size
                      else (image_processor.size['height'], image_processor.size['width'])),
    ToTensor(),
    Normalize(image_processor.image_mean, image_processor.image_std)
])


def transform(data):
    data['pixel_values'] = [transforms(img.convert('RGB')) for img in data['image']]
    del data['image']
    return data


food = food.with_transform(transform)

data_collator = DefaultDataCollator()

metric_ = evaluate.load('accuracy')


def metric(eval_res):
    logits, ref = eval_res
    pred = numpy.argmax(logits, axis=1)
    return metric_.compute(predictions=pred, references=ref)


model = AutoModelForImageClassification.from_pretrained(checkpoint, num_labels=len(labels),
                                                        id2label=id2label, label2id=label2id)
training_args = TrainingArguments(
    output_dir='image_class_trainer', remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

train_dataset = food["train"].shuffle(seed=42).select(range(100))
eval_dataset = food["test"].shuffle(seed=42).select(range(100))

trainer = Trainer(model=model, args=training_args, data_collator=data_collator,
                  train_dataset=train_dataset, eval_dataset=eval_dataset,
                  tokenizer=image_processor,
                  compute_metrics=metric)
trainer.train()
trainer.save_model('image_class2')
