import glob
import random
import shutil
import os

from TransferLearningInterpretation import config as cfg


ORIGINAL_PATH = "/project_antwerp/data/Fruit-Images-Dataset/"

TRAIN_PATH = ORIGINAL_PATH + "Training/"
TEST_PATH = ORIGINAL_PATH + "Test/"

classes = [s.split("/")[-1] for s in glob.glob(TRAIN_PATH + "*")]
random.seed(112)
sampled_classes = random.sample(classes, cfg.NUM_CLASSES)

if os.path.exists(cfg.DATA_PATH):
    shutil.rmtree(cfg.DATA_PATH)
os.mkdir(cfg.DATA_PATH)

for c in sampled_classes:
    shutil.copytree(TRAIN_PATH+c, cfg.DATA_TRAIN_PATH+c)
    shutil.copytree(TEST_PATH+c, cfg.DATA_TEST_PATH+c)
