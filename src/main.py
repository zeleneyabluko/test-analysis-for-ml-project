# this script will do the following:
# build confusion matrix for each model

from buildmatrix import getmarkedvalues

# Your model string
model = "model1"

# Call the function with the model string
result = getmarkedvalues(model)

# Do something with the result if needed
print(result)