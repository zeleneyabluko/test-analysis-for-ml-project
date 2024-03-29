# This script will do the following:
# Go through the files in /resources/logs folder and extract the file which belong to the selected model
# Build the array of marked values and the array of predicted values for this specific model
# Build the multiclass matrix
# Build the confusion matrix for each value

import os


def getmarkedvalues(model):

    work_dir = os.path.dirname(os.getcwd())
    folder_name = 'resources/logs'
    directory = os.path.join(work_dir, folder_name)
    model_files = [filename for filename in os.listdir(directory) if
                   filename.startswith('image_') and model in filename]
    # marked_values = []
    return model_files