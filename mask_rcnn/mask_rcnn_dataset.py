import os
import pandas as pd
import numpy as np
import PIL
import torch
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F


class MaskGalaxyDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_ids = dataframe['object_id'].unique()

    def __getitem__(self, idx):
        # load images and masks
        image_id = self.image_ids[idx]
        records = self.df[self.df['object_id'] == image_id]

        img_path = os.path.join(self.image_dir, 'pngs', str(image_id) + '.png')
        mask_path = os.path.join(self.image_dir, 'masks', str(image_id) + '_mask.png')
        img = read_image(img_path)
        mask = read_image(mask_path)

        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask and handle empty bounding boxes
        # boxes = masks_to_boxes(masks)
        boxes = records[['bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax']].values
        if np.isnan(boxes).all():
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes,dtype=torch.float32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # there is only one class
        # labels = torch.ones((num_objs,), dtype=torch.int64)
        labels_list = records[['labels']].values
        labels = torch.as_tensor(labels_list, dtype=torch.int64)
        labels = torch.squeeze(labels, 1)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target['boxes'] = tv_tensors.BoundingBoxes(boxes, format='XYXY', canvas_size=F.get_size(img))
        target['masks'] = tv_tensors.Mask(masks)
        target['labels'] = labels
        target['image_id'] = torch.tensor([image_id])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_ids)
