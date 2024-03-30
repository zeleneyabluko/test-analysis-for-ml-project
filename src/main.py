# this script will do the following:
# build confusion matrix for each model
import pandas as pd
from buildmatrix import getlistoffiles, getmarkedvalues, getpredictedvalues, buildmulticlassmatrix, definelabels,buildbinarymatrices

# Your model string
model = "model3"

# Call the function with the model string
files = getlistoffiles(model)
marked_values = getmarkedvalues(files)
predicted_values=getpredictedvalues(files)
labels = definelabels(marked_values, predicted_values)
cm = buildmulticlassmatrix(marked_values, predicted_values, labels)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print(cm)
buildbinarymatrices(marked_values, predicted_values, labels)
