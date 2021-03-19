'''
https://kgptalkie.com/human-activity-recognition-using-accelerometer-data/
https://github.com/laxmimerit/Human-Activity-Recognition-Using-Accelerometer-Data-and-CNN

'''

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
print(tf.__version__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from os import listdir
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_datasets():
    subjects = list()
    for folder in range(1,4):
        for filename in listdir('./data{0}'.format(str(folder))):
            if filename.endswith("csv"):
                values = csv.reader(open('./data{0}/'.format(str(folder)) + filename, "r"), delimiter = ",") # opens training data
                processedlist = []
                for row in values:
                    temp = [row[0],row[1],row[2],row[3],row[4]]
                    processedlist.append(temp)
                subjects.append(processedlist)
    return subjects

def plot_subject(subject):
    num = []
    x = []
    y = []
    z = []
    for row in subject:
        num.append(float(row[0]))
        x.append(float(row[1]))
        y.append(float(row[2]))
        z.append(float(row[3]))

    fig, axis = plt.subplots(3)
    axis[0].plot(num, x)
    axis[1].plot(num, y)
    axis[2].plot(num, z)
    plt.show()

def make_pandas(dataset):
    columns = ["time", "x", "y", "z", "label"]
    datasets = []
    for i in range(0,len(dataset)):
        datasets.append(pd.DataFrame(data = dataset[i], columns = columns))
    return datasets

def get_frames(df):
    frames = []
    labels = []
    for dataset in df:
        frame = []

        for i in range(0,len(dataset)):
            x = dataset['x'][i]
            y = dataset['y'][i]
            z = dataset['z'][i]
            frame.append([[int(x)], [int(y)], [int(z)]])

        frames.append(frame)
        #print(dataset["label"])
        labels.append(int(dataset['label'][0]))
    frames = np.asarray(frames)
    labels = np.asarray(labels)
    #print(labels[:10])
    return frames, labels

subjects = load_datasets()
datasets = make_pandas(subjects)

X, Y = get_frames(datasets)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

#print(len(X_train))
#print(X_train.shape)

X_train = X_train.reshape(len(X_train), 1000, 3, 1)
X_test = X_test.reshape(len(X_test), 1000, 3, 1)
print(X_train[0].shape)

model = Sequential()
model.add(Conv2D(16, (2, 2), activation = 'relu', input_shape = X_train[0].shape))
model.add(Dropout(0.1))

model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(6, activation='softmax'))

epochnum = 60

model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(X_train, y_train, epochs = epochnum, validation_data= (X_test, y_test), verbose=1)



def plot_learningCurve(history, epochs):
  # Plot training & validation accuracy values
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history.history['accuracy'])
  plt.plot(epoch_range, history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

plot_learningCurve(history, epochnum)

model.save("Model", save_format = "tf")