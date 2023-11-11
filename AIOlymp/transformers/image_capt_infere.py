from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import requests

url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/pokemon.png"
image = Image.open(requests.get(url, stream=True).raw)

checkpoint = 'image_capt'
image_processor = AutoProcessor.from_pretrained('microsoft/git-base')
model = AutoModelForCausalLM.from_pretrained(checkpoint)

inputs = image_processor(images=image, return_tensors="pt")
pixel_values = inputs.pixel_values

generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
generated_caption = image_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_caption)
