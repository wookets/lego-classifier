import torch

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
from load_dataset import load_dataset_from_folder

# Load the image
image_path = "brick.jpg"  # Replace this with your image path
image = Image.open(image_path)

# Preprocess the image using the feature extractor
feature_extractor = ViTImageProcessor.from_pretrained("./trained_image_classifier")
inputs = feature_extractor(image, return_tensors="pt")

# Get predictions
model = ViTForImageClassification.from_pretrained("./trained_image_classifier")
with torch.no_grad():
  outputs = model(**inputs)
  logits = outputs.logits
  predicted_class_idx = torch.argmax(logits, dim=1).item()

# Assuming you still have access to the dataset object from earlier to retrieve class names
brick_ds = load_dataset_from_folder('dataset')
class_names = brick_ds.features["label"].names
print(f"Predicted class: {class_names[predicted_class_idx]}")
print(outputs)
print(logits)
print(predicted_class_idx)