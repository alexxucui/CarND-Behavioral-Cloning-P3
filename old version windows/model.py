"""
Steering angel predictions model
"""

import json
import numpy as np
import csv				
import PIL
import cv2
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Activation, Lambda, ELU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

# Normal laps
f1 = open("./1/normal/driving_log.csv")
reader = csv.reader(f1)
files1 =list(reader)

#Recover laps
f2 = open("./1/recover/driving_log.csv")
reader = csv.reader(f2)
files2 =list(reader)

#Combine files
files = files1 + files2
files = np.asarray(files)


train_image_center = []

for file in files:
    img = cv2.imread(file[0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[60:140,40:280]
    img = cv2.resize(img, (200,66))
    train_image_center.append(img)

#Flip the images and reverse the steering angle (generate more data and make model less biased)

for file in files:
    img = cv2.imread(file[0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[60:140,40:280]
    img = cv2.resize(img, (200,66))
    img = cv2.flip(img, 1)
    train_image_center.append(img)  


train_image_center =  np.asarray(train_image_center)

train_steering = files[:,3].astype(float)

train_steering = np.append(train_steering, -train_steering)

batch_size = 32
nb_epoch = 5

row, col, ch = 66, 200, 3

model = Sequential()

model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch)))


model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='elu', name='Conv1'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='elu', name='Conv2'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='elu', name='Conv3'))
model.add(Dropout(.2))
model.add(Convolution2D(64, 3, 3, activation='relu', name='Conv4'))
model.add(Convolution2D(64, 3, 3, activation='relu', name='Conv5'))
model.add(Flatten())

model.add(Dense(1164, activation='relu', name='FC1'))
model.add(Dropout(.5))
model.add(Dense(100, activation='relu', name='FC2'))
model.add(Dense(50, activation='relu', name='FC3'))
model.add(Dense(10, activation='relu', name='FC4'))
model.add(Dense(1, name='output'))


#Have tried 0.001, 0.0001 and 0.00001 and use the best result
my_adam=Adam(lr=0.00001)
model.compile(loss="mse", optimizer=my_adam)

#Use 10% data to do validation (the validation loss is simiar to training loss -> no overfitting)
model.fit(train_image_center, 
          train_steering, 
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          verbose=1,
          shuffle = True,
          validation_split=0.1)

# save model

json_string = model.to_json()
with open('model.json','w') as f:
    json.dump(json_string,f,ensure_ascii=False)
model.save_weights('model.h5')
