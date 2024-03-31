# test-analysis-for-ml-project
This is a script for analysing the results of testing of three ML models
(analyzed logs can be found in the folder /resources/logs)

The systems under test are three computer vision models for military drones. They are meant to detect and classify different military objects (radars, warships, air defense systems, artillery etc.)

The script does the following:
- extracts information about marked objects vs predictions done my model from the logs;
- builds the multiclass confusion matrix for a selected model;
- builds the binary confusion matrix for each type of object.
