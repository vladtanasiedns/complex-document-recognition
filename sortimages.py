import os
import sys
import shutil

cwd = os.getcwd()

train_folder = os.path.join(cwd, "data/train")
test_folder = os.path.join(cwd, "data/test")
validate_folder = os.path.join(cwd, "data/validate")
images_path = os.path.join(cwd, "images")


def sortTrainingImages():
    # Moving training images into data folder into class folders for each class
    with open("./labels/train.txt", "r") as train_images:
        train_images_data = train_images.readlines()
        print("Sorting training images...")
        for img_data in train_images_data:
            img_data = img_data.split()
            path = os.path.join(images_path, img_data[0])
            cls = img_data[1]
            mvpath = os.path.join(train_folder, cls)
            if os.path.exists(mvpath) == False:
                os.mkdir(mvpath)
            shutil.copy(path, mvpath)
        print("Sorting of training images complete")


def sortValidationImages():
    # Moving validation images into data folder into folders for each class
    with open("./labels/val.txt", "r") as val_images:
        val_images_data = val_images.readlines()
        print("Sorting validation images...")
        for img_data in val_images_data:
            img_data = img_data.split()
            path = os.path.join(images_path, img_data[0])
            cls = img_data[1]
            mvpath = os.path.join(validate_folder, cls)
            if os.path.exists(mvpath) == False:
                os.mkdir(mvpath)
            shutil.copy(path, mvpath)
        print("Sorting of training images complete")


def sortTrainImages():
    # Moving validation images into data folder into folders for each class
    with open("./labels/test.txt", "r") as test_images:
        test_images_data = test_images.readlines()
        print("Sorting train images...")
        for img_data in test_images_data:
            img_data = img_data.split()
            path = os.path.join(images_path, img_data[0])
            cls = img_data[1]
            mvpath = os.path.join(test_folder, cls)
            if os.path.exists(mvpath) == False:
                os.mkdir(mvpath)
            shutil.copy(path, mvpath)
        print("Sorting of training images complete")
