from datasets import load_dataset
from transformers import pipeline

data = load_dataset('food101', split="validation[:10]")
data['image'][0].show()

pipe = pipeline('image-classification', './image_class2')
print(pipe(data['image'][0]))
