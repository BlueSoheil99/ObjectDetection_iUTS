import os

from src import fisheye_correction
from src import video_analysis
from src import pet_extractor

inputs = [] # list of file names
directory = os.fsencode("data/input")
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    inputs.append("data/input/" + filename)

print(inputs)

for i in range(len(inputs)):
    # fisheye correction goes here

    # run video analysis on input video
    model_output = video_analysis.video_analysis(inputs[i], i) # in format (df, results, (fps, resolution))

    # run pet_extractor on the dataframe
    pet_output = pet_extractor.extract_pet(model_output[0], sec_cutoff=4, traj_cutoff=0.9, mag_cutoff=50, valid_classes=["bus","truck","car"])

    # run the video formulator on results object, take ids as input
    for output in pet_output:
        video_analysis.specific_output([output[0], output[1]], model_output[1], model_output[2], i)
