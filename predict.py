# Imports
from unicodedata import category
import tensorflow as tf
tf.get_logger().setLevel('WARNING')
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import time
from PIL import Image
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import utilities as ut

parser = argparse.ArgumentParser(description='Flowers Image Classifier')
parser.add_argument('path')
parser.add_argument('model')
parser.add_argument('--top_k')
parser.add_argument('--category_names') 
args = parser.parse_args()

image_path = args.path
image = Image.open(image_path)

top_k = int(args.top_k)
if top_k is None: 
    top_k = 1

model = tf.keras.models.load_model(args.model, custom_objects={'KerasLayer':hub.KerasLayer})
   
probs, classes = ut.predict(image_path, model, top_k)

print("\nClasses:", classes)
print("\nProbabilities:", probs)

if args.category_names is not None:
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
    print("\nFlowernames:")
    flower_names = []
    for c in classes[0]:
        flower_names.append(class_names[str(c+1)])
    print(flower_names)
    label = flower_names[0]
    print("\nMost probably it is a", label)
