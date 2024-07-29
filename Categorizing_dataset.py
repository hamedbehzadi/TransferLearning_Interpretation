import numpy as np
import shutil
import os
import random
import glob

main_add = './project_antwerp/dataset/15-Scene/'
target_add = './project_antwerp/dataset/15_Scene_categorized/'
classes_name = ['00','01','02','03','04','05','06','07','08','09','11','12','13','14']

for class_name in classes_name:
    class_dir = main_add + class_name +'/'
    class_images_add = os.listdir(class_dir)
    random.shuffle(class_images_add)
    train_length = int(0.8*len(class_images_add))
    test_length = int(len(class_images_add) - train_length)
    train_add = class_images_add[0:train_length]
    test_add = class_images_add[train_length:]
    for image_add in train_add:
        if '.ipynb_checkpoints' in image_add:
            continue
        if not os.path.exists(target_add + 'train/'+class_name):
            os.mkdir(target_add + 'train/'+class_name)
        shutil.copy(class_dir+image_add,target_add+'train/'+class_name+'/')
    for image_add in test_add:
        if '.ipynb_checkpoints' in image_add:
            continue
        if not os.path.exists(target_add + 'val/'+class_name):
            os.mkdir(target_add + 'val/'+class_name)
        shutil.copy(class_dir+image_add,target_add+'val/'+class_name+'/')
