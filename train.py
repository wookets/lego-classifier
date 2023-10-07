import os
import datasets

from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from tensorflow import keras
from tensorflow.keras import layers

from load_dataset import load_dataset_from_folder

dataset_path = './dataset' 
model_id = 'google/vit-base-patch16-224-in21k'

# 1. Load the dataset
brick_ds = load_dataset_from_folder('dataset')
test_ds = load_dataset_from_folder('testset')
# img_class_labels = brick_ds.features["label"].names
# print(img_class_labels)

# 3. Load feature extractor for ViT
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# 4. Preprocess the images
def preprocess_function(examples):
  return feature_extractor(examples['img'], return_tensors="pt")

encoded_dataset = brick_ds.map(preprocess_function, batched=True)
encoded_testset = test_ds.map(preprocess_function, batched=True)

# 5. Load and prepare the model
num_labels = len(set(brick_ds['label']))
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=num_labels)

# 6. Training
training_args = TrainingArguments(
  output_dir = './results',
  per_device_train_batch_size = 8,
  num_train_epochs = 3,
  evaluation_strategy = 'epoch',
  logging_dir = './logs',
  logging_steps = 10,
  save_steps = 10,
  eval_steps = 10,
  save_total_limit = 2,
  push_to_hub = False,
)

trainer = Trainer(
  model = model,
  args = training_args,
  train_dataset = encoded_dataset,
  eval_dataset = encoded_testset,
)

trainer.train()

# 7. Save the trained model
model.save_pretrained("./trained_image_classifier")
feature_extractor.save_pretrained("./trained_image_classifier")

print("Training complete. Model saved to './trained_image_classifier'")