import typing
import os
import logging
import argparse

import omegaconf
import hydra
# https://hydra.cc/docs/configure_hydra/intro/#accessing-the-hydra-config
from hydra.core.hydra_config import HydraConfig

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

from zoobot.shared import schemas
from galaxy_datasets.shared import label_metadata

from galaxy_datasets import transforms
import gz3d_pytorch_model, pytorch_datamodule



@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(config : omegaconf.DictConfig) -> None:

    pl.seed_everything(config.random_state)

    save_dir = HydraConfig.get().run.dir

    on_local = os.path.isdir('/Users/user/')
    if on_local:
        base_dir = '/Users/user/repos/zoobot-3d/'
    else:
        base_dir = '/share/nas2/walml/galaxy_zoo/segmentation/'

    debug = config.debug
    if debug or on_local:
        config.max_galaxies = 500
        config.gz3d_galaxies_only = True
        config.spiral_galaxies_only = True
        config.oversampling_ratio = 1
        config.max_epochs = 2
        config.patience = 2
        config.image_size = 128
        config.batch_size = 32
        config.num_workers = 1
        config.accelerator = 'cpu'
        config.precision = '32-true'
        config.devices = 'auto'
        config.strategy = 'auto'
        
        torch.set_float32_matmul_precision('medium')

    if config.schema_name == 'desi_dr5':
        schema = schemas.decals_dr5_ortho_schema  # new - just DR5
    elif config.schema_name == 'desi_all':
        schema = schemas.decals_all_campaigns_ortho_schema
    elif config.schema_name == 'desi_and_gz2':
        schema = desi_and_gz2_schema()
    else:
        raise ValueError(config.schema_name)

    wandb.config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )
    wandb_logger = WandbLogger(project='zoobot-3d', log_model=False, config=wandb.config)
    # wandb.init(project='zoobot-3d')
    config = wandb.config

    # df = pd.read_parquet(base_dir + 'data/gz3d_and_gz_desi_master_catalog.parquet')
    df = pd.read_parquet(base_dir + 'data/gz3d_and_desi_master_catalog.parquet')  # now includes GZ2 also
    if config.spiral_galaxies_only:
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
    if config.gz3d_galaxies_only:
        df = df.query('spiral_mask_exists')
        assert len(df) > 0

    # hide votes for galaxies with masks
    logging.info(df[df['spiral_mask_exists']][schema.label_cols[0]])
    df[schema.label_cols] = df[schema.label_cols].fillna(0).astype(int)
    # where cond False (i.e. where spiral_mask_exists=True), replace with other (0)
    df[schema.label_cols] = df[schema.label_cols].where(~df['spiral_mask_exists'], 0)
    logging.info(df[df['spiral_mask_exists']][schema.label_cols[0]])

    logging.info(f'Galaxies in catalog: {len(df)}')

    if config.max_galaxies is not None:
        df = df[:config.max_galaxies]
        logging.info(f'Galaxies after cut: {len(df)}')

    
    train_catalog, hidden_catalog = train_test_split(df, test_size=0.3, random_state=args.random_state)
    val_catalog, test_catalog = train_test_split(hidden_catalog, test_size=0.2/0.3, random_state=args.random_state)

    log_every_n_steps = min(int(len(train_catalog) / config.batch_size), 100)

    # oversampling
    if config.oversampling_ratio > 1:
        logging.info('Using oversampling')
        spiral_masked_galaxies = train_catalog[train_catalog['spiral_mask_exists']]
        train_catalog = pd.concat(
            [train_catalog] + [spiral_masked_galaxies]*(config.oversampling_ratio-1)
        )
        # and shuffle again
        train_catalog = train_catalog.sample(frac=1, random_state=args.random_state).reset_index(drop=True)

    
    model = gz3d_pytorch_model.ZooBot3D(
        input_size=config.image_size,
        n_classes=2,  # spiral segmap, bar segmap
        output_dim=len(schema.label_cols),
        question_index_groups=schema.question_index_groups,
        use_vote_loss=config.use_vote_loss,
        use_seg_loss=config.use_seg_loss,
        seg_loss_weighting=config.seg_loss_weighting
    )

    datamodule = pytorch_datamodule.SegmentationDataModule(
        train_catalog=train_catalog,
        val_catalog=val_catalog,
        test_catalog=test_catalog,
        label_cols=schema.label_cols,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        transform=default_segmentation_transforms(resize_after_crop=config.image_size)
    )
    datamodule.setup('fit')

    callbacks = [
        EarlyStopping(
            monitor=config.loss_to_monitor,
            patience=config.patience,
            check_finite=True,
            verbose=True
        ),
        ModelCheckpoint(dirpath=save_dir, monitor=config.loss_to_monitor)
    ]
    # use this obj so we can log the string above
    if config.strategy == 'ddp':
        logging.info('Using DDP strategy')
        
        strategy_obj = DDPStrategy(
            accelerator=config.accelerator,
            parallel_devices=[torch.device(f"cuda:{i}") for i in range(config.devices)],
            find_unused_parameters=True
        )
    else:
        strategy_obj = config.strategy

    trainer = pl.Trainer(
        # accelerator=config.accelerator,
        # devices=config.devices,
        max_epochs=config.max_epochs,
        precision=config.precision,
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

    trainer.test(model, datamodule)

def get_jpg_loc(row, base_dir):
    if row['relative_desi_jpg_loc'] == None:
        return row['galahad_jpg_loc']
    else:
        return base_dir + row['relative_desi_jpg_loc']


def desi_and_gz2_schema():


    question_answer_pairs = {}
    question_answer_pairs.update(label_metadata.decals_all_campaigns_ortho_pairs)
    question_answer_pairs.update(label_metadata.gz2_ortho_pairs)

    dependencies = {}
    dependencies.update(label_metadata.decals_ortho_dependencies)
    dependencies.update(label_metadata.gz2_ortho_dependencies)

    # print(question_answer_pairs)
    # print(dependencies)

    schema = schemas.Schema(question_answer_pairs, dependencies)

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

if __name__ == '__main__':


    logging.basicConfig(level=logging.INFO)
    train()

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--save-dir', dest='save_dir', default='results/models', type=str)
#     parser.add_argument('--max-galaxies', dest='max_galaxies', default=None, type=int)
#     parser.add_argument('--gz3d-galaxies-only', dest='gz3d_galaxies_only', default=False, action='store_true')
#     parser.add_argument('--spiral-galaxies-only', dest='spiral_galaxies_only', default=False, action='store_true')
#     parser.add_argument('--use-vote-loss', dest='use_vote_loss', default=False, action='store_true')
#     parser.add_argument('--use-seg-loss', dest='use_seg_loss', default=False, action='store_true')
#     parser.add_argument('--seg-loss-weighting', dest='seg_loss_weighting', default=100., type=float)
#     parser.add_argument('--oversampling', dest='oversampling_ratio', default=1, type=int)
#     parser.add_argument('--debug', dest='debug',
#                         default=False, action='store_true')
#     parser.add_argument('--seed', dest='random_state', default=42, type=int)
#     args = parser.parse_args()

#     main(args)
