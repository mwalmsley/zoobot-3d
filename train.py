import typing
import os
import logging
import argparse

import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from sklearn.model_selection import train_test_split
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2

from zoobot.shared.schemas import Schema
from galaxy_datasets.shared import label_metadata

from galaxy_datasets import transforms
import gz3d_pytorch_model, pytorch_datamodule


def desi_and_gz2_schema():


    question_answer_pairs = {}
    question_answer_pairs.update(label_metadata.decals_all_campaigns_ortho_pairs)
    question_answer_pairs.update(label_metadata.gz2_ortho_pairs)

    dependencies = {}
    dependencies.update(label_metadata.decals_ortho_dependencies)
    dependencies.update(label_metadata.gz2_ortho_dependencies)

    # print(question_answer_pairs)
    # print(dependencies)

    schema = Schema(question_answer_pairs, dependencies)

    return schema


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
        gz3d_galaxies_only = True
        spiral_galaxies_only = True
        oversampling_ratio = 1
        max_epochs = 2
        patience = 2
        image_size = 128
        batch_size = 32
        num_workers = 1
        accelerator = 'cpu'
        precision = '32-true'
        log_every_n_steps = 10
        devices = 'auto'
        strategy = 'auto'
    else:
        max_galaxies = None
        # gz3d_galaxies_only = True
        gz3d_galaxies_only = True
        # spiral_galaxies_only = False
        spiral_galaxies_only = True
        # oversampling_ratio = 10
        oversampling_ratio = 1
        # log_every_n_steps = 100
        log_every_n_steps = 9
        max_epochs = 1000
        # patience = 50
        patience = 5
        image_size = 224
        batch_size = 256  # 2xA100 at mixed precision
        num_workers = 12
        accelerator = 'gpu'
        devices = 2
        precision = '16-mixed'
        
        strategy = 'ddp'
        torch.set_float32_matmul_precision('medium')

    seg_loss_weighting = 100
    # seg_loss_weighting = 0  # TODO warning
    # loss_to_monitor = 'validation/epoch_total_loss:0'
    loss_to_monitor = 'validation/epoch_seg_loss:0'
    

 
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
        'precision': precision,
        'strategy': strategy,
        'gz3d_galaxies_only': gz3d_galaxies_only,
        'spiral_galaxies_only': spiral_galaxies_only,
        'oversampling_ratio': oversampling_ratio,
        'loss_to_monitor': loss_to_monitor
    }
    wandb_logger = WandbLogger(project='zoobot-3d', log_model=False, config=config)
    wandb.init(project='merger', config=config)  # args will be ignored by existing logger
    wandb_config = wandb.config

    # df = pd.read_parquet(base_dir + 'data/gz3d_and_gz_desi_master_catalog.parquet')
    df = pd.read_parquet(base_dir + 'data/gz3d_and_desi_master_catalog.parquet')  # now includes GZ2 also
    if wandb_config.spiral_galaxies_only:
        # these are PREDICTED fractions, hence no _dr12, _gz2, etc
        # consistent across GZ2/DESI (nice)
        df = df[df['smooth-or-featured_featured-or-disk_fraction'] > 0.5]
        df = df[df['disk-edge-on_yes_fraction'] < 0.5]
        df = df[df['has-spiral-arms_yes_fraction'] > 0.5]

    logging.info(df['relative_spiral_mask_loc'].iloc[0])

    # adjust paths for base_dir
    df['spiral_mask_loc'] = df['relative_spiral_mask_loc'].astype(str).apply(lambda x: base_dir + x)
    df['bar_mask_loc'] = df['relative_bar_mask_loc'].astype(str).apply(lambda x: base_dir + x)

    # df['relative_desi_jpg_loc'] = df['relative_desi_jpg_loc'].astype(str)
    df['desi_jpg_loc'] = df.apply(lambda x: get_jpg_loc(x, base_dir), axis=1)
    # print(df['relative_desi_jpg_loc'].sample(5))
    # print(df['desi_jpg_loc'].sample(5))

    logging.info(df['spiral_mask_loc'].iloc[0])

    logging.info('Check paths')
    # TODO should precalculate
    df['spiral_mask_exists'] = df['spiral_mask_loc'].apply(os.path.isfile)
    if wandb_config.gz3d_galaxies_only:
        df = df.query('spiral_mask_exists')
        assert len(df) > 0

    logging.info(f'Galaxies in catalog: {len(df)}')

    if wandb_config.max_galaxies is not None:
        df = df[:max_galaxies]
        logging.info(f'Galaxies after cut: {len(df)}')

    
    train_catalog, hidden_catalog = train_test_split(df, test_size=0.3, random_state=args.random_state)
    val_catalog, test_catalog = train_test_split(hidden_catalog, test_size=0.2/0.3, random_state=args.random_state)

    schema = desi_and_gz2_schema()

    # oversampling
    if wandb_config.oversampling_ratio > 1:
        logging.info('Using oversampling')
        spiral_masked_galaxies = train_catalog[train_catalog['spiral_mask_exists']]
        train_catalog = pd.concat(
            [train_catalog] + [spiral_masked_galaxies]*(wandb_config.oversampling_ratio-1)
        )
        # and shuffle again
        train_catalog = train_catalog.sample(frac=1, random_state=args.random_state).reset_index(drop=True)

    
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

    callbacks = [
        EarlyStopping(
            monitor=wandb_config.loss_to_monitor,
            patience=wandb_config.patience,
            check_finite=True,
            verbose=True
        ),
        ModelCheckpoint(dirpath=args.save_dir, monitor=wandb_config.loss_to_monitor)
    ]
    # use this obj so we can log the string above
    if wandb_config.strategy == 'ddp':
        logging.info('Using DDP strategy')
        
        strategy_obj = DDPStrategy(
            accelerator=wandb_config.accelerator,
            parallel_devices=[torch.device(f"cuda:{i}") for i in range(wandb_config.devices)],
            find_unused_parameters=True
        )
    else:
        strategy_obj = strategy

    trainer = pl.Trainer(
        # accelerator=wandb_config.accelerator,
        # devices=wandb_config.devices,
        max_epochs=wandb_config.max_epochs,
        precision=wandb_config.precision,
        logger=wandb_logger,
        log_every_n_steps=log_every_n_steps,
        strategy=strategy_obj,
        callbacks=callbacks,
        # some extra args to try to avoid rare nans
        gradient_clip_val=0.5,
        detect_anomaly=True,
        num_nodes=1
    )

    trainer.fit(model, datamodule)

    # trainer.test(model, datamodule)

def get_jpg_loc(row, base_dir):
    if row['relative_desi_jpg_loc'] == None:
        return row['galahad_jpg_loc']
    else:
        return base_dir + row['relative_desi_jpg_loc']


if __name__ == '__main__':

    main()
