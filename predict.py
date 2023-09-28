import os
import logging

# import h5py
import torch
import pytorch_lightning as pl
import pandas as pd
import omegaconf
import hydra
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

import pytorch_datamodule
import gz3d_pytorch_model
# import train

@hydra.main(version_base=None, config_path="conf", config_name="config")
def predict(config : omegaconf.DictConfig) -> None:
    pl.seed_everything(config.random_state)

    on_local = os.path.isdir('/Users/user/')
    if on_local:
        base_dir = '/Users/user/repos/zoobot-3d/'
    else:
        base_dir = '/share/nas2/walml/galaxy_zoo/segmentation/'

    df = pd.read_parquet(base_dir + 'data/test_catalog.parquet')
    logging.info(df['spiral_mask_loc'].iloc[0])
    df['spiral_mask_exists'] = df['spiral_mask_loc'].apply(os.path.isfile)
    assert any(df['spiral_mask_exists'])
    df = df.query('spiral_mask_exists')[:8].reset_index(drop=True)
    df['spiral_mask_loc'] = ''

    # model = pl.load_from_checkpoint(config.checkpoint_path)
    # model.freeze()

    checkpoint_path = base_dir + 'outputs/run_1695899881.3925836/epoch=93-step=1880.ckpt'

    model = gz3d_pytorch_model.ZooBot3D.load_from_checkpoint(checkpoint_path, map_location='cpu')
    model.eval()

    datamodule = pytorch_datamodule.SegmentationDataModule(
        predict_catalog=df,
        # batch_size=config.batch_size,
        # num_workers=config.num_workers,
        num_workers=4,
        batch_size=64,
        # transform=train.default_segmentation_transforms(resize_after_crop=config.image_size)
        transform=predict_transform(resize_after_crop=config.image_size)
    )
    datamodule.setup('predict')

    trainer = pl.Trainer(
        # accelerator=config.accelerator,
        accelerator=config.accelerator,
        devices=1,
        max_epochs=config.max_epochs,
        # precision=config.precision,
        logger=False,
        num_nodes=1
    )
    preds = trainer.predict(model, datamodule)

    # preds is list of batches
    # each batch is tensor of segmap output (batch_size, n_classes, 224, 224)
    print(len(preds))
    print(preds[0].shape)
    preds = torch.concat(preds, dim=0)  # stick batches back together
    print(preds.shape)
    save_dir = os.path.dirname(checkpoint_path) + '/predictions/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    for galaxy_index in range(len(preds)):
        galaxy = df.iloc[galaxy_index]
        spiral_mask = preds[galaxy_index][0]
        bar_mask = preds[galaxy_index][1]
        print(spiral_mask.min(), spiral_mask.max())
        Image.fromarray(
            (spiral_mask.numpy() * 255).astype('uint8')
            ).save(save_dir + f'{galaxy["dr8_id"]}_spiral_pred_viz.png')
        Image.fromarray(
            (bar_mask.numpy() * 255).astype('uint8')
            ).save(save_dir + f'{galaxy["dr8_id"]}_bar_pred_viz.png')
        # these will line up with loaded images, provided you also apply the center crop (optionally with albumentations itself)
        # some precision in uncertainty is lost

def predict_transform(resize_after_crop=224):
    # training transform uses crop scale of 0.7 to 0.8
    # so crop to original size * 0.75
    # NOW 0.6-0.7, original*0.65
    transforms_to_apply = [
        A.CenterCrop(height=int(424*0.65), width=int(424*0.65)),
        # then resize as normal
        A.Resize(height=resize_after_crop, width=resize_after_crop, interpolation=1),
        A.ToFloat(max_value=255.),  # TODO remove, need different max value for each
        ToTensorV2()  # channels first, torch convention
    ]

    return A.Compose(
        transforms_to_apply,
        additional_targets={'spiral_mask': 'image', 'bar_mask': 'image'}
    )

if __name__ == '__main__':
    predict()
