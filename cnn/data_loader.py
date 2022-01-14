import os
import PIL

import torch
from torch.utils.data import Dataset
import imgaug.augmenters as ia

class BaseDataset(Dataset):
    """
    dataset 디렉토리 구조

    train - truck - img1.jpg .....
          - suv - img1.jpg .....
          - car - img1.jpg .....
          - van - img1.jpg .....
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.label_names = list()

        self.img_list = list()
        self.label_list = list()
        for index, dir_name in enumerate(sorted(os.listdir(self.root_dir))):
            dir_path = os.path.join(self.root_dir, dir_name)

            if os.path.isdir(dir_path):
                self.label_names.append((index, dir_name))

                for img_name in sorted(os.listdir(dir_path)):
                    if os.path.splitext(img_name)[-1] in [".jpg", ".png", ".jpeg"]:
                        img_path = os.path.join(dir_path, img_name)

                        self.img_list.append(img_path)
                        self.label_list.append(index)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        sample = dict()

        img_path = self.img_list[idx]
        label = self.label_list[idx]

        # ==============Annotation 구간=================
        # opencv 및 PIL library 를 사용 하여 이미지 annotation 진행
        # transform 만 사용해도 무관
        img = PIL.Image.open(img_path).convert('RGB')
        # imgaug 적용 부분
        if self.is_train:
            seq = ia.Sequential([
                ia.pillike.Autocontrast((5, 20), per_channel=True),
                ia.Fliplr(0.5),
                ia.MotionBlur(k=5),
                ia.Sharpen(alpha=(0, 1), lightness=(0.75, 2)),
                ia.ChangeColorTemperature((1100, 10000))
            ])
            img = seq.augment_image(img)

        img = PIL.Image.fromarray(img)

        # main 에서 진행한 transform 적용 (annotation)
        if self.transform:
            image = self.transform(img)
        # ==============================================

        sample['image'] = image
        sample['label'] = label
        sample['image_name'] = os.path.basename(img_path)

        return sample

    def get_label_info(self):
        return self.label_names