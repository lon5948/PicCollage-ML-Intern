import csv
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def read_responses(response_path: str):
    data = dict()
    with open(response_path, newline='') as f:
        rows = csv.DictReader(f)
        data = {row['id']: row['corr'] for row in rows}
    return data


def read_images(data_dir: str, data: dict):
    x_data = np.zeros((len(data), 128, 128, 3), dtype=np.uint8)
    y_data = np.zeros((len(data)), dtype=np.float32)
    
    for i, id in enumerate(tqdm(data.keys())): 
        image_path = os.path.join(data_dir, "images", f"{id}.png")
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 128))
        x_data[i] = image
        y_data[i] = data[id]
    
    return x_data, y_data


def preprocessing(data_dir: str):
    response_path = os.path.join(data_dir, "responses.csv")
    data = read_responses(response_path)
    x_data, y_data = read_images(data_dir, data)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    return x_train, y_train, x_test, y_test
