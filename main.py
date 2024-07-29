import os

from TransferLearningInterpretation import compute_similarities
from TransferLearningInterpretation import config as cfg
from TransferLearningInterpretation import train
from TransferLearningInterpretation import generate_visualisations
from TransferLearningInterpretation.dataset import CustomActivationData, DataSingle, DataSplit
from TransferLearningInterpretation.utils import create_if_not_exist

import torch
print(torch.__version__)

TRAIN = True
RF_EXTRACT = False
SIMILARITY = False
VISUALISATION = False


print(f"Run utilizing model {cfg.MODEL_NAME} and dataset {cfg.DATASET_NAME}")

if TRAIN:
    print(f"Loading dataset")
    data = None
    if cfg.DATASET_NAME == "AwA2" or cfg.DATASET_NAME == "15Scene":
        data = DataSingle(cfg.DATA_PATH)
    elif cfg.DATASET_NAME == "Fruits":
        data = DataSplit(cfg.DATA_TRAIN_PATH, cfg.DATA_TEST_PATH,cfg)
    else:
        RuntimeError("Invalid dataset selected.")

    print("Training model")
    create_if_not_exist(cfg.MODEL_OUTPUT_PATH)

    if not os.path.exists(cfg.MODEL_OUTPUT_PATH+"pretrained.pth"):
        train.save(cfg.BASE_MODEL, cfg.MODEL_OUTPUT_PATH+"pretrained.pth")

    training_framework = train.Train(cfg.BASE_MODEL, data, cfg.DEVICE)
    training_framework.fit(
        epochs=cfg.EPOCHS, checkpoints=cfg.CHECKPOINTS, output_path=cfg.MODEL_OUTPUT_PATH)

    print("training loss:", training_framework.train_loss_history)
    print("validation loss:", training_framework.val_loss_history)

if RF_EXTRACT:
    print("Extracting activation maps")
    data = CustomActivationData(cfg.RF_DATA_PATH)

    if cfg.MODEL_NAME == "resnet50":
        import resnet50_extract
        extractor = resnet50_extract.Resnet50Extract(data,model)
    elif cfg.MODEL_NAME == "densenet":
        import densenet_extract
        extractor = densenet_extract.DensenetExtract(data)
    else:
        import model_extract
        extractor = model_extract.ModelExtract(data)

    extractor.extract()

if SIMILARITY:
    print("Computing similarities")
    compute_similarities.run(cfg)

if VISUALISATION:
    print("Creating visualisations")
    generate_visualisations.run()
