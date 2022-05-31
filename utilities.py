import tensorflow as tf
from PIL import Image
import numpy as np 

def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    return image.numpy()

def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    expected_image = np.expand_dims(image, axis=0)
    probs = model.predict(expected_image)
    probs, classes = tf.nn.top_k(probs, k=top_k)
    probs = probs.numpy()
    classes = classes.numpy()
    return probs, classes