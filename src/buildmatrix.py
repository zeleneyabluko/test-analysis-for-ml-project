# This script will do the following:
# Go through the files in /resources/logs folder and extract the file which belong to the selected model
# Build the array of marked values and the array of predicted values for this specific model
# Build the multiclass matrix
# Build the confusion matrix for each value

import os
import json
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

work_dir = os.path.dirname(os.getcwd())
logs_dir = 'resources/logs'
directory = os.path.join(work_dir, logs_dir)

def getlistoffiles(model):
    model_files = [filename for filename in os.listdir(directory) if
                   filename.startswith('image_') and model in filename]
    return model_files

def getmarkedvalues(model_files):
    marked_values = []
    for file in model_files:
        path = os.path.join(directory, file)
        with open(path, "r") as f:
            marked_value = json.load(f)['marked_object']
            marked_values.append(marked_value)
    marked_values = np.array(marked_values)
    return marked_values


def getpredictedvalues(model_files):
    predicted_values = []
    for file in model_files:
        path = os.path.join(directory, file)
        with open(path, "r") as f:
            predicted_value = json.load(f)['predicted_object']
            predicted_values.append(predicted_value)
    predicted_values = np.array(predicted_values)
    return predicted_values

def definelabels(marked_values,  predicted_values):
    labels = np.unique(np.concatenate((marked_values, predicted_values)))
    return labels
def buildmulticlassmatrix(marked_values, predicted_values, labels):

    matrix = confusion_matrix(marked_values, predicted_values, labels=labels)
    return matrix

def buildbinarymatrices(multiclass_matrix):



