# lego-classifier

This will take an image input (of a lego) and classify it by the type (plate, brick, etc) of lego. 

## Getting Started

The model, `trained_image_classifier`, is already trained based on the `dateset` and `testset`. This was done using the `train.py` script. 

For now, You should be able to drop an image `brick.jpg` into the root folder and run `python infer.py`.

The dataset and testset can be found on an s3 bucket - TODO. 