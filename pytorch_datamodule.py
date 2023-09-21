import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd

import pytorch_dataset

class SegmentationDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_catalog: pd.DataFrame,
            val_catalog: pd.DataFrame,
            test_catalog=None,
            label_cols=None,
            transform=None,
            batch_size: int=32,
            num_workers: int=8,
            prefetch_factor: int=4
        ):
        super().__init__()

        self.train_catalog = train_catalog
        self.val_catalog = val_catalog
        self.test_catalog = test_catalog
        self.label_cols = label_cols
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset = pytorch_dataset.SegmentationGalaxyDataset(
            catalog=self.train_catalog,
            label_cols=self.label_cols,
            transform=self.transform
            )
            self.val_dataset = pytorch_dataset.SegmentationGalaxyDataset(
                catalog=self.val_catalog,
                label_cols=self.label_cols,
                transform=self.transform
            )
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            assert self.test_catalog is not None
            self.test_dataset = pytorch_dataset.SegmentationGalaxyDataset(
                catalog=self.test_catalog,
                label_cols=None,
                transform=self.transform
            )

        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)

# def collate_segmaps(batch):

if __name__ == '__main__':


    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose(
        [
            A.CenterCrop(250, 250),
            A.ToFloat(max_value=255.),  # otherwise will still be uint8
            # TODO normalise over dataset?
            ToTensorV2()
        ],
        additional_targets={'spiral_mask': 'image', 'bar_mask': 'image'}
    )

    df = pd.read_csv('/Users/user/repos/zoobot-3d/data/gz3d_and_gz_desi_matches.csv')[:100]

    # temp, until I update master df
    df['segmap_json_loc'] = df['local_gz3d_fits_loc'].str.replace('/fits_gz/', '/segmaps/', regex=False).str.replace('.fits.gz', '.json', regex=False)
    df['local_desi_jpg_loc'] = df.apply(lambda x: f'data/desi/jpg/{x["brickid"]}/{x["brickid"]}_{x["objid"]}.jpg', axis=1)

    
    datamodule = SegmentationDataModule(
        train_catalog=df[:2],
        val_catalog=df[2:4],
        test_catalog=df[4:6],
        batch_size=2,
        num_workers=1,
        transform=transform
    )
    datamodule.setup('fit')

    for batch in datamodule.train_dataloader():
        print(batch)
        break
