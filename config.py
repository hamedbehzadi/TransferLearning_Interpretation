import torch
import torch.nn as nn
import torchvision.models as models
import timm

import os
from TransferLearningInterpretation import utils

# Project config
BASE_PATH = "./project_antwerp/TransferLearningInterpretation/"
OUTPUT_PATH = BASE_PATH + "models/"

SKIP_EXISTING = True

# Dataset config
INPUT_SIZE = [224, 224]

# Run selection parameters
_override = True
_settings = dict()

if not _override:
    _dataset_file_select = "run_select.pbz2"

    if os.path.exists(BASE_PATH + _dataset_file_select):
        _settings = utils.load_and_decompress_pickle(
            BASE_PATH + _dataset_file_select)
    else:
        _settings = {"dataset": 0, "model": 0}

    # Prepare values for next run, iterate over models and then dataset
    _new_settings = dict()
    _new_settings["model"] = (_settings["model"] + 1) % 4
    _new_settings["dataset"] = (
                                       _settings["dataset"] + int(_new_settings["model"] == 0)) % 3
    utils.save_and_compress_pickle(
        BASE_PATH + _dataset_file_select, _new_settings)
else:
    # Set override
    """
    dataset:
        0: Fruits
        1: 15Scenes
        2: AwA2
        3: ImageNet
    model:
        0: vgg19
        1: alexnet
        2: densenet
        3: resnet50
        4: ViT
    """
    _settings = {"dataset": 0, "model": 4}

if _settings["dataset"] == 0:
    # Fruits dataset
    DATASET_NAME = "Fruits"
    DATA_PATH = BASE_PATH + "data/Fruits/"
    DATA_TRAIN_PATH = DATA_PATH + "Training/"
    DATA_TEST_PATH = DATA_PATH + "Test/"
    NUM_CLASSES = 20
elif _settings["dataset"] == 1:
    # 15 Scenes dataset
    DATASET_NAME = "15Scene"
    DATA_PATH = "./project_ghent/dataset/15-Scene/"
    NUM_CLASSES = 15
elif _settings["dataset"] == 3:
    # 15 Scenes dataset
    DATASET_NAME = "ImageNet"
    DATA_PATH = ""
    NUM_CLASSES = 1000
else:
    # AwA2 animal dataset
    DATASET_NAME = "AwA2"
    DATA_PATH = "./project_ghent/dataset/AwA2/Animals_with_Attributes2/JPEGImages/"
    NUM_CLASSES = 50

OUTPUT_PATH = OUTPUT_PATH + DATASET_NAME + "/"

# Model config
if _settings["model"] == 0:
    MODEL_NAME = "vgg19"
    BASE_MODEL = models.vgg19(pretrained=True, progress=False)
    BASE_MODEL.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    BASE_MODEL.name = MODEL_NAME
    INSPECT_LAYERS_FEATURE = [3, 8, 17, 26, 35]
    INSPECT_LAYERS_CLASSIFIER = []
    VISUALISATION_LAYER = [26, 35]
    HEATMAP_LAYER = [3, 8, 17]
elif _settings["model"] == 1:
    MODEL_NAME = "alexnet"
    BASE_MODEL = models.alexnet(pretrained=True, progress=False)
    BASE_MODEL.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    BASE_MODEL.name = MODEL_NAME
    INSPECT_LAYERS_FEATURE = [1, 4, 11]
    INSPECT_LAYERS_CLASSIFIER = []
    VISUALISATION_LAYER = [11]
    HEATMAP_LAYER = [1, 4]
elif _settings["model"] == 2:
    MODEL_NAME = "densenet"
    BASE_MODEL = models.densenet121(pretrained=True, progress=False)
    BASE_MODEL.classifier = nn.Linear(1024, NUM_CLASSES)
    BASE_MODEL.name = MODEL_NAME
    INSPECT_LAYERS_FEATURE = []
    INSPECT_LAYERS_CLASSIFIER = []
    VISUALISATION_LAYER = []
    HEATMAP_LAYER = []
elif _settings["model"] == 3:
    MODEL_NAME = "resnet50"
    BASE_MODEL = models.resnet50(pretrained=True, progress=False)
    BASE_MODEL.fc = nn.Linear(2048, NUM_CLASSES)
    BASE_MODEL.name = MODEL_NAME
    INSPECT_LAYERS_FEATURE = [25, 54, 97, 119]
    INSPECT_LAYERS_CLASSIFIER = []
    VISUALISATION_LAYER = [25, 54, 97, 119]
    HEATMAP_LAYER = [25, 54]
elif _settings["model"] == 4:
    MODEL_NAME = "ViT"
    BASE_MODEL = timm.create_model('vit_base_patch16_224', pretrained=True,num_classes=NUM_CLASSES)
    BASE_MODEL.fc = nn.Linear(2048, NUM_CLASSES)
    BASE_MODEL.name = MODEL_NAME
    INSPECT_LAYERS_FEATURE = [25, 54, 97, 119]
    INSPECT_LAYERS_CLASSIFIER = []
    VISUALISATION_LAYER = [25, 54, 97, 119]
    HEATMAP_LAYER = [25, 54]

# logging path
MODEL_OUTPUT_PATH = OUTPUT_PATH + MODEL_NAME + "/"

# Training config
EPOCHS = 15
CHECKPOINTS = [1, 3, 5, 7, 10, 15]
LEARNING_RATE = 5e-5#0.0001
TRAINING_BATCHSIZE = 32#196
VALIDATION_BATCHSIZE = 32#196  # 196 / 48

# Devices
DEVICE_STR = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(DEVICE_STR)
DEVICE_CPU = torch.device("cpu")

# Activation maps
RF_DATA_PATH = BASE_PATH + "data/activations/" + DATASET_NAME + "/"
RF_OUTPUT_PATH = BASE_PATH + "activation_maps/" + \
                 DATASET_NAME + "/" + MODEL_NAME + "/"

# Similarities
SIM_OUTPUT_PATH = BASE_PATH + "similarities/" + \
                  DATASET_NAME + "/" + MODEL_NAME + "/"

WANDB_LOGGING = False

if WANDB_LOGGING:
    try:
        # WanDB configs
        import wandb

        wandb.init(project="Thesis", entity="lderoeck")
        wandb.config = {
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": TRAINING_BATCHSIZE,
        }
    except:
        print("No wandb install found, switching to terminal logging.")


        class wandb:
            def __init__(self) -> None:
                pass

            def log(obj: object):
                print(obj)

            def watch(obj: object):
                pass
else:
    class wandb:
        def __init__(self) -> None:
            pass

        def log(obj: object):
            print(obj)

        def watch(obj: object):
            pass
