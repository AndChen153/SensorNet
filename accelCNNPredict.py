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

samples_import = ["./data/data4.csv", "./data/data9.csv","./data/data8.csv"]

def load_datasets(datasets):
    subjects = list()
    for filename in datasets:
        values = csv.reader(open(filename, "r"), delimiter = ",") # opens training data
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
    frames = np.asarray(frames)
    return frames

columns = ["time", "x", "y", "z", "label"]
subjects = load_datasets(samples_import)
datasets = []
for i in range(0,len(subjects)):
    datasets.append(pd.DataFrame(data = subjects[i], columns = columns))

samples_predict = get_frames(datasets)
samples_predict = samples_predict.reshape(len(samples_import), 1000, 3, 1)

predictions = model.predict(samples_predict)

classes = np.argmax(predictions, axis = 1)
print(classes)