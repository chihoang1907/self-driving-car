import os
import pandas as pd

directory = "dataset/"
dir_inputs = directory + "JPEGImages/"
dir_outputs = directory + "Annotations/"

file_data = "data/data.csv"

for root, dirs, files in os.walk(dir_inputs):
    inputs = [os.path.join(root, f) for f in files]

for root, dirs, files in os.walk(dir_outputs):
    outputs = [os.path.join(root, f) for f in files]

pd.DataFrame({"Input":inputs,"Output":outputs}).to_csv(file_data, index= False)