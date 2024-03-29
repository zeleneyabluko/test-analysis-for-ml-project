# this script will do the following:
# build confusion matrix for each model

from buildmatrix import getlistoffiles, getmarkedvalues, getpredictedvalues

# Your model string
model = "model1"

# Call the function with the model string
files = getlistoffiles(model)
marked_values = getmarkedvalues(files)
predicted_values=getpredictedvalues(files)

# Do something with the result if needed
print(predicted_values)