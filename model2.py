from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    raw_image = Image.open(image_path).convert('RGB')

    inputs = processor(raw_image, return_tensors="pt")

    captions = model.generate(**inputs)

    generated_caption = processor.decode(captions[0], skip_special_tokens=True)
    return generated_caption


