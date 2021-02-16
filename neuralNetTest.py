# TensorFlow and tf.keras
#import tensorflow as tf
#from tensorflow import keras

# Helper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
import csv

def load_dataset():
    subjects = list()
    for filename in listdir('.'):
        if filename.endswith("csv"):
            values = csv.reader(open(filename, "r"), delimiter = ",") # opens training data
            subjects.append(values)
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

subjects = load_dataset()
print(subjects[0])
plot_subject(subjects[0])
