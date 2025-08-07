"""
main.py

This script should be called from terminal in order to:
- run a computer vision model on a given input video
- train a computer vision model on a given dataset
- open a video

Author: <NAME>
Date: <DATE>
"""

import argparse
import os

import yaml

from src import fisheye_correction
from src import video_analysis
from src import pet_extractor
from src import model_trainer
from src.utils import video_player


def load_config(config_path:str) -> dict:
    """
    Load YAML configuration file.
    # TODO if such a function is used in other files, make a read_config.py in utils folder

    :param config_path: Path to the YAML config file.
    :return: Parsed configuration as a dictionary.
    """
    print(f'--- Reading {config_path} config file')
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    keys = config.keys()
    for key in keys:
        print(f'\t {key} : {config[key]}')
    return config


def run_model(configs:dict):
    """

    :param configs: a dictionary that contains necessary configs for running a model
    :return:
    """
    model_path = configs['model']
    input_folder = configs['data']['input_folder']
    output_folder = configs['data']['output_folder']

    inputs = [] # list of file names
    directory = os.fsencode(input_folder)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        inputs.append(os.path.join(input_folder, filename))


    for i in range(len(inputs)):
        # fisheye correction goes here

        # run video analysis on input video
        model_output = video_analysis.video_analysis(video_path=inputs[i],
                                                     model=model_path,
                                                     output_path=output_folder,
                                                     id=i)
        # in format (df, results, (fps, resolution))

        # run pet_extractor on the dataframe
        pet_output = pet_extractor.extract_pet(model_output[0], sec_cutoff=4,
                                               traj_cutoff=0.9, mag_cutoff=50,
                                               valid_classes=["bus","truck","car"])

        # run the video formulator on results object, take ids as input
        for output in pet_output:
            video_analysis.specific_output(output_folder,[output[0], output[1]],
                                           model_output[1], model_output[2], i)



def create_parser(defaults):
    """
    Create the argument parser so that arguments can be passed as command line arguments.
    :param defaults:
    :return: an "argparse.ArgumentParser" object.
    """
    # Create the parser object
    parser = argparse.ArgumentParser(
        description="A program that runs computer vision tracking on videos"
                    " in the data/input folder, with a focus on vehicle detection.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # Add optional arguments
    parser.add_argument(
        "-a", "--action",
        choices=["detect", "train", "open_video"],
        default=defaults['action'],
        help=f"The action to be performed by this script. default: {defaults['action']}"
    )
    parser.add_argument(
        "-m", "--model",
        default=defaults['model'],
        help=f"The ID or path of the model to use. Default is {defaults['model']}."
    )  # TODO document what exactly model 0 and 1 are.

    parser.add_argument(
        '-c', '--config',
        default=defaults['config'],
        help='Path to the config YAML file'
    )
    return parser


if __name__ == "__main__":
    default_values = {'action':'detect',
                      'model':'resources/models/1.pt',
                      'config':'resources/configs/config.yaml'}

    # Create the parser object
    parser = create_parser(default_values)
    # Parse the arguments from the command line
    args = parser.parse_args()
    # Load configs
    configs = load_config(args.config)

    # Call the main function with the parsed arguments
    if args.action == "detect":
        run_model(configs)

    elif args.action == "train":
        model_path = configs['model']
        training_config = configs['training']
        model_trainer.train(model_path, training_config)

    elif args.action == "open_video":
        print('opening video!')
        folder = configs["data"]["input_folder"]
        file = configs["data"]["input_file"]
        video_player.open_video(os.path.join(folder, file))
