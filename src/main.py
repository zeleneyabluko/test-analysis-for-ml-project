# this script will do the following:
# build confusion matrix for each model
import pandas as pd
from buildmatrix import getlistoffiles, getmarkedvalues, getpredictedvalues, buildmulticlassmatrix, definelabels

# Your model string
model = "model1"

# Call the function with the model string
files = getlistoffiles(model)
marked_values = getmarkedvalues(files)
predicted_values=getpredictedvalues(files)
labels = definelabels(marked_values, predicted_values)
cm = buildmulticlassmatrix(marked_values, predicted_values, labels)
print(pd.DataFrame(cm, index=labels, columns=labels))
