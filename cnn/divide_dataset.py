import os
import random
from shutil import copy2

if __name__ == '__main__':
    data_dir = 'crawl/버스'
    train_dir = 'data/train'
    val_dir = 'data/val'

    val_rate = 0.1

    for label in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)

        image_list = []
        for filename in sorted(os.listdir(label_dir)):  # we sort image list
            ext = os.path.splitext(filename)[-1]
            if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                image_path = os.path.join(label_dir, filename)
                image_list.append(image_path)

        # shuffle image list
        random.Random(123).shuffle(image_list)

        val_image_count = int(len(image_list) * val_rate)

        val_image_list = image_list[:val_image_count]
        train_image_list = image_list[val_image_count:]

        # create train label dir
        train_label_dir = os.path.join(train_dir, label)
        val_label_dir = os.path.join(val_dir, label)
        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(val_label_dir, exist_ok=True)

        # copy images
        for image_path in train_image_list:
            copy2(image_path, os.path.join(train_label_dir, os.path.basename(image_path)))

        for image_path in val_image_list:
            copy2(image_path, os.path.join(val_label_dir, os.path.basename(image_path)))