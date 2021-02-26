# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from scipy import stats
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

def plot_activity(subject):
    time = []
    x = []
    y = []
    z = []
    for row in subject:
        time.append(float(row[0]))
        x.append(float(row[1]))
        y.append(float(row[2]))
        z.append(float(row[3]))
        activity = row[4]

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 7), sharex=True)
    plot_axis(ax0, time, x, 'X-Axis')
    plot_axis(ax1, time, y, 'Y-Axis')
    plot_axis(ax2, time, z, 'Z-Axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()

def plot_axis(ax, x, y, title):
    ax.plot(x, y, 'g')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


plot_activity(subjects[0])
plot_activity(subjects[5])
