from transformers import AutoProcessor, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import evaluate
from textwrap import wrap
from matplotlib import pyplot
import numpy

data = load_dataset('lambdalabs/pokemon-blip-captions')
data = data['train'].train_test_split(test_size=0.1)


def plot_images(images, captions):
    pyplot.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = pyplot.subplot(1, len(images), i + 1)
        caption = captions[i]
        caption = "\n".join(wrap(caption, 12))
        pyplot.title(caption)
        pyplot.imshow(images[i])
        pyplot.axis("off")


# sample_images_to_visualize = [numpy.array(data['train'][i]["image"]) for i in range(5)]
# sample_captions = [data['train'][i]["text"] for i in range(5)]
# plot_images(sample_images_to_visualize, sample_captions)
# pyplot.show()

checkpoint = 'microsoft/git-base'
image_processor = AutoProcessor.from_pretrained(checkpoint)


def transform(example_batch):
    images = [x for x in example_batch["image"]]
    captions = [x for x in example_batch["text"]]
    inputs = image_processor(images=images, text=captions, padding="max_length")
    inputs.update({"labels": inputs["input_ids"]})
    return inputs


data['train'] = data['train'].shuffle().select(range(10)).with_transform(transform)
data['test'] = data['test'].shuffle().select(range(3)).with_transform(transform)

model = AutoModelForCausalLM.from_pretrained(checkpoint)
metric_ = evaluate.load('wer')


def metric(eval_pred):
    logits, labels = eval_pred
    predicted = logits.argmax(-1)
    decoded_labels = image_processor.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions = image_processor.batch_decode(predicted, skip_special_tokens=True)
    score = metric_.compute(predictions=decoded_predictions, references=decoded_labels)
    return {"wer_score": score}


model_name = checkpoint.split("/")[1]

training_args = TrainingArguments(
    output_dir=f"image_capt_trainer",
    learning_rate=5e-5,
    num_train_epochs=50,
    # fp16=True,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    save_total_limit=3,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    logging_steps=50,
    remove_unused_columns=False,
    label_names=["labels"],
    load_best_model_at_end=True,
)

trainer = Trainer(model, training_args, train_dataset=data['train'], eval_dataset=data['test'],
                  # tokenizer=image_processor,
                  compute_metrics=metric)
trainer.train()
trainer.save_model('image_capt')
