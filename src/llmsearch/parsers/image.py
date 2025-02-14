from typing import Dict, List, Union
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, GPT2Tokenizer
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor
from PIL import Image
from pathlib import Path


# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Initialize the model and feature extractor
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def preprocess_image(image):
    # For instance, resize the image if required by your model
    # image = image.resize((target_width, target_height))
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    return image

# Function to generate an image caption
def generate_image_caption(image):
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

    # Generate captions
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4, early_stopping=True)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

def image_captioner(
    path: Union[str, Path], *args, **additional_splitter_setting
) -> List[Dict[str, str]]:
    image = Image.open(path)
    image = preprocess_image(image)
    return [{"text":generate_image_caption(image), "metadata":{}}]

if __name__ == "__main__":
    # Load an image
    print(image_captioner("/Users/tamer/PycharmProjects/llm-search/notebooks/sample_docs/Screenshot 2024-02-25 at 14.27.19.png"))