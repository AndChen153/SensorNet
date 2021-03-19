'''
https://www.machinecurve.com/index.php/2020/02/21/how-to-predict-new-samples-with-your-keras-model/
'''

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

model = tf.keras.models.load_model('./Model')
# Check its architecture
model.summary()

def load_datasets():
    subjects = list()
    for filename in listdir('./testingdata'):
        if filename.endswith("csv"):
            values = csv.reader(open('./data{0}/'.format(str(folder)) + filename, "r"), delimiter = ",") # opens training data
            processedlist = []
            for row in values:
                temp = [row[0],row[1],row[2],row[3],row[4]]
                processedlist.append(temp)
            subjects.append(processedlist)
    return subjects

def get_frames(df):
    frames = []
    for dataset in df:
        frame = []
        for i in range(0,len(dataset)):
            x = dataset['x'][i]
            y = dataset['y'][i]
            z = dataset['z'][i]
            
            frame.append([int(x), int(y), int(z)])
        frames.append(frame)
        labels.append(int(dataset['label'][0]))
    frames = np.asarray(frames)
    labels = np.asarray(labels)
    return frames,labels

columns = ["time", "x", "y", "z"]
subjects = load_datasets()
datasets = []
for i in range(0,len(subjects)):
    datasets.append(pd.DataFrame(data = subjects[i], columns = columns))

samples_predict, samples_answers = get_frames(datasets)
samples_predict = samples_predict.reshape(len(samples_import), 1000, 3, 1)

predictions = model.predict(samples_predict)

classes = np.argmax(predictions, axis = 1)
correct = 0
for i in range(0,len(classes))
    if classes[i] == samples_answers[i]:
        correct += 1
print("amount correct: " , correct/len(classes))