import typing
import os

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl

import albumentations as A
from albumentations.pytorch import ToTensorV2

from zoobot.shared.schemas import decals_all_campaigns_ortho_schema

from galaxy_datasets import transforms
import gz3d_pytorch_model, pytorch_datamodule

# lazy copy of the below, but with additional_targets=
# https://github.com/mwalmsley/galaxy-datasets/blob/main/galaxy_datasets/transforms.py#L6
def default_segmentation_transforms(
    crop_scale_bounds=(0.7, 0.8),
    crop_ratio_bounds=(0.9, 1.1),
    resize_after_crop=224, 
    pytorch_greyscale=False
    ) -> typing.Dict[str, typing.Any]:

    transforms_to_apply = [
        A.Lambda(name="RemoveAlpha", image=transforms.RemoveAlpha(), always_apply=True)
    ]

    if pytorch_greyscale:
        transforms_to_apply += [
            A.Lambda(
                name="ToGray", image=transforms.ToGray(reduce_channels=True), always_apply=True
            )
        ]

    transforms_to_apply += [
        A.Rotate(limit=180, interpolation=1,
                    always_apply=True, border_mode=0, value=0),
        A.RandomResizedCrop(
            height=resize_after_crop,  # after crop resize
            width=resize_after_crop,
            scale=crop_scale_bounds,  # crop factor
            ratio=crop_ratio_bounds,  # crop aspect ratio
            interpolation=1,
            always_apply=True
        ),  # new aspect ratio
        A.VerticalFlip(p=0.5),
        # new here, for the byte masks
        A.ToFloat(max_value=255.),
        ToTensorV2()  # channels first, torch convention
    ]

    return A.Compose(
        transforms_to_apply,
        additional_targets={'spiral_mask': 'image', 'bar_mask': 'image'}
    )


def main():

    df = pd.read_csv('/Users/user/repos/zoobot-3d/data/gz3d_and_gz_desi_master_catalog.csv')
    df = df[df['smooth-or-featured_featured-or-disk_fraction'] > 0.5]
    df = df[df['disk-edge-on_yes_fraction'] < 0.5]
    df = df[df['has-spiral-arms_yes_fraction'] > 0.5]
    df['spiral_mask_exists'] = df['local_spiral_mask_loc'].apply(os.path.isfile)
    df = df.query('spiral_mask_exists')
    print(len(df))

    schema = decals_all_campaigns_ortho_schema

    image_size = 128
    model = gz3d_pytorch_model.ZooBot3D(input_size=image_size, question_index_groups=schema.question_index_groups)

    datamodule = pytorch_datamodule.SegmentationDataModule(
        train_catalog=df[:100],
        val_catalog=df[200:300],
        test_catalog=df[400:500],
        label_cols=schema.label_cols,
        batch_size=2,
        num_workers=1,
        transform=default_segmentation_transforms(resize_after_crop=128)
    )
    datamodule.setup('fit')

    trainer = pl.Trainer(
        accelerator='cpu',
        max_epochs=2,
        precision='32-true'
    )

    trainer.fit(model, datamodule)

if __name__ == '__main__':

    main()
