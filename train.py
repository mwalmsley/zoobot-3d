import typing
import os
import logging
import argparse

import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import wandb
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



    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', dest='save_dir', default='results/models', type=str)
    parser.add_argument('--debug', dest='debug',
                        default=False, action='store_true')
    parser.add_argument('--seed', dest='random_state', default=42, type=int)
    args = parser.parse_args()

    pl.seed_everything(args.random_state)

    on_local = os.path.isdir('/Users/user/')
    if on_local:
        base_dir = '/Users/user/repos/zoobot-3d/'
    else:
        base_dir = '/share/nas2/walml/galaxy_zoo/segmentation/'

    debug = args.debug
    if debug or on_local:
        max_galaxies = 500
        max_epochs = 2
        patience = 2
        image_size = 128
        batch_size = 32
        num_workers = 1
        accelerator = 'cpu'
        precision = '32-true'
        log_every_n_steps = 10
        devices = 'auto'
    else:
        max_galaxies = None
        max_epochs = 1000
        patience = 5
        image_size = 224
        batch_size = 128
        num_workers = 12
        accelerator = 'gpu'
        devices = 2
        precision = '16-mixed'
        log_every_n_steps = 100
        torch.set_float32_matmul_precision('medium')

    seg_loss_weighting = 100


 
    config = {
        'debug': debug,
        'max_epochs': max_epochs,
        'patience': patience,
        'image_size': image_size,
        'seg_loss_weighting': seg_loss_weighting,
        'max_galaxies': max_galaxies,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'accelerator': accelerator,
        'devices': devices,
        'precision': precision
    }
    wandb_logger = WandbLogger(project='zoobot-3d', log_model=False, config=config)
    wandb.init(project='merger', config=config)  # args will be ignored by existing logger
    wandb_config = wandb.config

    df = pd.read_csv(base_dir + 'data/gz3d_and_gz_desi_master_catalog.csv')
    df = df[df['smooth-or-featured_featured-or-disk_fraction'] > 0.5]
    df = df[df['disk-edge-on_yes_fraction'] < 0.5]
    df = df[df['has-spiral-arms_yes_fraction'] > 0.5]

    logging.info(df['local_spiral_mask_loc'].iloc[0])

    # adjust paths for base_dir
    df['local_spiral_mask_loc'] = df['local_spiral_mask_loc'].apply(lambda x: base_dir + x)
    df['local_bar_mask_loc'] = df['local_bar_mask_loc'].apply(lambda x: base_dir + x)
    df['local_desi_jpg_loc'] = df['local_desi_jpg_loc'].apply(lambda x: base_dir + x)

    logging.info(df['local_spiral_mask_loc'].iloc[0])

    df['spiral_mask_exists'] = df['local_spiral_mask_loc'].apply(os.path.isfile)

    # df = df.query('spiral_mask_exists')
    assert len(df) > 0, df['local_desi_jpg_loc'].iloc[0]
    logging.info(f'Galaxies in catalog: {len(df)}')

    if wandb_config.max_galaxies is not None:
        df = df[:max_galaxies]
        logging.info(f'Galaxies after cut: {len(df)}')

    
    train_catalog, hidden_catalog = train_test_split(df, test_size=0.3, random_state=args.random_state)
    val_catalog, test_catalog = train_test_split(hidden_catalog, test_size=0.2/0.3, random_state=args.random_state)

    schema = decals_all_campaigns_ortho_schema

    
    model = gz3d_pytorch_model.ZooBot3D(
        input_size=wandb_config.image_size,
        n_classes=2,  # spiral segmap, bar segmap
        output_dim=len(schema.label_cols),
        question_index_groups=schema.question_index_groups,
        seg_loss_weighting=wandb_config.seg_loss_weighting
    )

    datamodule = pytorch_datamodule.SegmentationDataModule(
        train_catalog=train_catalog,
        val_catalog=val_catalog,
        test_catalog=test_catalog,
        label_cols=schema.label_cols,
        batch_size=wandb_config.batch_size,
        num_workers=wandb_config.num_workers,
        transform=default_segmentation_transforms(resize_after_crop=wandb_config.image_size)
    )
    datamodule.setup('fit')

    # or for seg loss specifically:
    # validation/epoch_seg_loss:0 

    callbacks = [
        EarlyStopping(monitor='validation/epoch_loss:0', patience=wandb_config.patience),
        ModelCheckpoint(dirpath=args.save_dir, monitor='validation/epoch_loss:0')
    ]

    trainer = pl.Trainer(
        accelerator=wandb_config.accelerator,
        devices=wandb_config.devices,
        max_epochs=wandb_config.max_epochs,
        precision=wandb_config.precision,
        logger=wandb_logger,
        log_every_n_steps=log_every_n_steps,
        strategy='auto',
        callbacks=callbacks
    )

    trainer.fit(model, datamodule)

    # trainer.test(model, datamodule)


if __name__ == '__main__':

    main()
