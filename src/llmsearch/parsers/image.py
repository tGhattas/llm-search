from typing import List, Union
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, GPT2Tokenizer
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor
import matplotlib.pyplot as plt
from PIL import Image

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
) -> List[str]:
    image = Image.open(path)
    image = preprocess_image(image)
    return [generate_image_caption(image)]