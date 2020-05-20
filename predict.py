#Necessary Imports
from PIL import Image
import argparse
import tensorflow as tf
import tensorflow_hub as hub
import time
import numpy as np
import matplotlib.pyplot as plt
import json

#To Avoid Warnings
import warnings
warnings.filterwarnings('ignore')

#Loggings
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#Command Line Postional and Optional Arguments
parser = argparse.ArgumentParser(description = 'Udacity Project')
parser.add_argument('image_path')
parser.add_argument('reload_model')
parser.add_argument('--top_k', type = int, default = 5)
parser.add_argument('--category_names', default = 'label_map.json')

#Command Line Arguments received are mapped to the respective variables
image_path = parser.parse_args().image_path
saved_model = parser.parse_args().reload_model
top_k = parser.parse_args().top_k
category_names = parser.parse_args().category_names

#Loading Class Names of the Dataset
with open('label_map.json', 'r') as f:
    class_names = json.load(f)
  
#Reloaded the Model from Part-1 of the Project
model = tf.keras.models.load_model(saved_model,custom_objects={'KerasLayer':hub.KerasLayer})

#Function to Preprocess the image which is fed into the network 
def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image,(224,224))
    image = tf.cast(image, tf.float32)
    image /= 255
    image = image.numpy()
    
    return image

#Function which make the top_k predictions (probabilities and labels)
def predict_(image_path,model,top_k):
    im = Image.open(image_path)
    image = np.asarray(im)
    image = process_image(image)
    predictions = model.predict(np.expand_dims(image,axis=0))
    probs,classes = tf.math.top_k(predictions, k=top_k, sorted=True, name=None)
    probs=probs.numpy()[0]
    classes=(classes.numpy()+1)[0]
    classes=[class_names[str(value)] for value in classes]
    
    return probs,classes

#Calling the Predict_ to make predictions on the image specified in image_path through Command Line
probs,classes=predict_(image_path,model,top_k)

#Printing the top_k Probabilities and Labels which were predicted by the model
print('The Top {} Probabilities and Predictions are as follows:'.format(top_k))
for each in range(0,len(probs)):
    print("[{}]: Prediction says that, Input Image is '{}' with a Probability of {}".format(each+1,classes[each], probs[each]))


 
