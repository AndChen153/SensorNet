'''
https://kgptalkie.com/human-activity-recognition-using-accelerometer-data/
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
    for filename in listdir('.'):
        if filename.endswith("csv"):
            values = csv.reader(open(filename, "r"), delimiter = ",") # opens training data
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

subjects = load_datasets()
#plot_subject(subjects[0])
columns = ["time", "x", "y", "z", "label"]

dataset = pd.DataFrame(data = subjects[0], columns = columns)
dataset2 = pd.DataFrame(data = subjects[1], columns = columns)
dataset3 = pd.DataFrame(data = subjects[2], columns = columns)
#print(dataset.head())
#print(dataset.shape)
#print(dataset.info())


def get_frames(df):
    frames = []
    labels = []
    for dataset in df:
        frame = []
        for i in range(0,len(dataset)):
            x = dataset['x'][i]
            y = dataset['y'][i]
            z = dataset['z'][i]
            
            frame.append([x,y,z])
        frames.append(frame)
        labels.append(dataset["label"][0])
    frames = np.asarray(frames)
    labels = np.asarray(labels)
    return frames, labels

X_train, y_train = get_frames([dataset])
X_test, y_test = get_frames([dataset3])

X_train = X_train.reshape(1, 1000, 3, 1)
X_test = X_test.reshape(1, 1000, 3, 1)
print(X_train[0].shape, X_test[0].shape)

model = Sequential()
model.add(Conv2D(16, (2, 2), activation = 'relu', input_shape = X_train[0].shape))
model.add(Dropout(0.1))

model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(6, activation='softmax'))

model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(X_train, y_train)

'''
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

#plot_learningCurve(history, 10)


from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


y_pred = model.predict_classes(X_test)

mat = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat=mat, class_names=label.classes_, show_normed=True, figsize=(7,7))'''