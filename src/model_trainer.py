"""
model_trainer.py

"""
from ultralytics import YOLO

def train(model_path, training_configs):
    """
    based on: https://docs.ultralytics.com/modes/train/ for YOLO models
    :param model_path: Where the pre-trained model is stored
    :param training_configs: Configs needed for training a new model. Includes new_model_name, dataset, epochs, etc.
    :return:
    """
    print('training modules is called')

    # Load a pretrained model
    model = YOLO(model_path)

    # Train the model #todo modify this
    results = model.train(data=training_configs['dataset'],
                          epochs=training_configs['epoch'],
                          device=training_configs['device'])
    # results = model.train(data=training_configs['dataset'], epochs=2)
