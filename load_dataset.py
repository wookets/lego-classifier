import os
import datasets

def load_dataset_from_folder(dataset_dir):
  # parse out class names
  class_names = os.listdir(dataset_dir)
  # define datasets features
  features = datasets.Features({ "img": datasets.Image(), "label": datasets.features.ClassLabel(names=class_names) })
  # a place to store results
  img_data_files = []
  label_data_files = []
  # load images into list for creation
  for img_class in os.listdir(dataset_dir):
    if img_class == '.DS_Store':
      continue
    for img in os.listdir(os.path.join(dataset_dir, img_class)):
      path = os.path.join(dataset_dir, img_class, img)
      img_data_files.append(path)
      label_data_files.append(img_class)
  # create dataset
  ds = datasets.Dataset.from_dict({ "img": img_data_files, "label": label_data_files }, features=features)
  return ds