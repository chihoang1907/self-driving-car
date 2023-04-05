import os
import pandas as pd

# https://www.kaggle.com/datasets/sakshaymahna/kittiroadsegmentation
directory = "dataset/training/"
dir_inputs = directory + "image_2/"
dir_outputs = directory + "gt_image_2/"

file_data = "data/data.csv"

for root, dirs, files in os.walk(dir_inputs):
    inputs = [os.path.join(root, f) for f in files]

for root, dirs, files in os.walk(dir_outputs):
    outputs = []
    for f in files:
        if "road" in f:
            outputs.append(os.path.join(root, f))

pd.DataFrame({"Input":inputs,"Output":outputs}).to_csv(file_data, index= False)