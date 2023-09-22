import json
import logging
import io
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmap_utils


from galaxy_datasets.pytorch import galaxy_dataset

class SegmentationGalaxyDataset(galaxy_dataset.GalaxyDataset):

    def __init__(self, catalog: pd.DataFrame, label_cols=None, transform=None, target_transform=None) -> None:
        self.catalog = catalog
        self.label_cols = label_cols
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index: int):

        outputs = {}

        galaxy = self.catalog.iloc[index]

        # load the image into memory
        image_loc = galaxy['local_desi_jpg_loc']
        try:
            image = np.array(galaxy_dataset.load_img_file(image_loc))
            # HWC PIL image, 0-255 uint8
        except Exception as e:
            logging.critical('Cannot load {}'.format(image_loc))
            raise e

        # new bit - load segmaps
        # with open(galaxy['segmap_json_loc'], 'r') as f:
        #     segmaps = json.load(f)
        #     segmap_dict = {}
        #     for segmap_name, marks_by_users in segmaps.items():
        #         segmap_image = construct_segmap_image(galaxy, marks_by_users)
        #         segmap_dict[segmap_name] = segmap_image
        # or static mode
        segmap_dict = {}
        spiral_mask_loc = galaxy['local_spiral_mask_loc']
        if os.path.isfile(spiral_mask_loc):
            segmap_dict['spiral_mask'] = np.expand_dims(np.array(galaxy_dataset.load_img_file(spiral_mask_loc)), 2)
            # print(segmap_dict['spiral_mask'].shape)
        else:
            # need to always return something so that batch elements will be stackable
            segmap_dict['spiral_mask'] = np.zeros((image.shape[0], image.shape[1], 1)).astype(np.uint8)
        bar_mask_loc = galaxy['local_bar_mask_loc']
        if os.path.isfile(bar_mask_loc):
            segmap_dict['bar_mask'] = np.expand_dims(np.array(galaxy_dataset.load_img_file(bar_mask_loc)), 2)
        else:
            segmap_dict['bar_mask'] = np.zeros((image.shape[0], image.shape[1], 1)).astype(np.uint8)

        if self.transform:
            try:
                transformed = self.transform(image=image, **segmap_dict)
            except Exception as e:
                logging.critical('Cannot transform {}, {}, {}'.format(image_loc, spiral_mask_loc, bar_mask_loc))
                logging.critical(image.shape)
                logging.critical(segmap_dict['spiral_mask'].shape)
                logging.critical(segmap_dict['bar_mask'].shape)
                raise e
            
            image = transformed['image']
            # everything else is the transformed segmap dict
            del transformed['image']
            segmap_dict = transformed
            # segmap_dict = dict([(k, v) for k, v in transformed.items() if k != 'image'])
    
        outputs['image'] = image
        # e.g. spiral_mask, bar_mask (inplace)
        outputs.update(segmap_dict)

        if self.label_cols is not None:
            # load the labels. If no self.label_cols, key will not exist
            label = galaxy_dataset.get_galaxy_label(galaxy, self.label_cols)    

            # I never use this tbh
            if self.target_transform:
                label = self.target_transform(label)
            # no effect on mask labels
            outputs['label_cols'] = label

        
        return outputs




if __name__ == '__main__':

        
        df = pd.read_csv('/Users/user/repos/zoobot-3d/data/gz3d_and_gz_desi_matches.csv')

        # temp, until I update master df
        df['segmap_json_loc'] = df['local_gz3d_fits_loc'].str.replace('/fits_gz/', '/segmaps/', regex=False).str.replace('.fits.gz', '.json', regex=False)
        df['local_desi_jpg_loc'] = df.apply(lambda x: f'data/desi/jpg/{x["brickid"]}/{x["brickid"]}_{x["objid"]}.jpg', axis=1)

        galaxy = df.iloc[32]

        # # minimal test of construct_segmap_image
        # with open(galaxy['segmap_json_loc'], 'r') as f:
        #     segmaps = json.load(f)
        # spiral_marks = segmaps['spiral']
        # mask_im = construct_segmap_image(galaxy, spiral_marks)

        # desi_im = Image.open(galaxy['local_desi_jpg_loc'])

        # fig, ax = plt.subplots()
        # ax.imshow(desi_im, alpha=.5)
        # ax.imshow(mask_im, alpha=.5)
        # # plt.show()
        # # plt.savefig('temp_latest.png', bbox_inches='tight', pad_inches=0., facecolor='purple')

        # # test of dataloader (without transforms)
        # dataset = SegmentationGalaxyDataset(df)
        # image, segmap_dict = dataset[32]
        # fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(10, 3))
        # ax0.imshow(segmap_dict['spiral'])
        # ax1.imshow(segmap_dict['bar'])
        # ax2.imshow(image)
        # plt.show()

        # # test of dataloader with manual albumentations transform
        # dataset = SegmentationGalaxyDataset(df)
        # image, segmap_dict = dataset[32]

        # transform = A.Compose(
        #     [
        #         A.RandomCrop(300, 300)
        #     ],
        #     # can't do dynamically, need to know how many inputs to expect before transform
        #     additional_targets={'spiral_mask': 'image', 'bar_mask': 'image'}
        # )
        # transformed = transform(image=image, spiral_mask=segmap_dict['spiral'], bar_mask=segmap_dict['bar'])

        # fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(10, 3))
        # ax0.imshow(transformed['spiral_mask'])
        # ax1.imshow(transformed['bar_mask'])
        # ax2.imshow(transformed['image'])  # default target keyed as image
        # plt.show()

        # test of dataloader with albumentations transform
        transform = A.Compose(
            [
                A.RandomCrop(300, 300),
                # ToTensorV2()  # would use this if really PyTorch, sets channels first
            ],
            # can't do dynamically, need to know how many inputs to expect before transform
            additional_targets={'spiral_mask': 'image', 'bar_mask': 'image'}
        )
        dataset = SegmentationGalaxyDataset(df, transform=transform)
        outputs = dataset[32]

        fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(10, 3))
        ax0.imshow(outputs['spiral_mask'])
        ax1.imshow(outputs['bar_mask'])
        ax2.imshow(outputs['image'])  # default target keyed as image
        plt.show()


"""

import wcs_utils

# work out the WCS of the segmap
segmap_wcs = wcs_utils.get_wcs_assuming_image_size(
    center_ra=galaxy['ra_manga'],  # TODO
    center_dec=galaxy['dec_manga'],
    # GZ3D numbers
    center_pixel_index=525/2.,
    arcsec_per_pixel=0.099
)

# work out the WCS of the DESI jpg image
target_image_wcs = wcs_utils.get_wcs_assuming_image_size(
    center_ra=galaxy['ra'],
    center_dec=galaxy['dec'],
    center_pixel_index=424/2, # target jpg is 424x424,  
    arcsec_per_pixel=galaxy['est_dr5_pixscale']
)

# plot the segmap pixels on the target image WCS frame
# can I do this directly on pixels without imshow? probably yes
# segmap_wcs.pixel_to_world

fig = plt.figure()
# image will appear on desi projection
fig = plt.figure(figsize=(1, 1))
ax = plt.Axes(fig, [0., 0., 1., 1.])

ax = plt.subplot(projection=target_image_wcs)

arcsec_width = 10
# arcsec_width = 52
# arcsec_width = galaxy['est_dr5_pixscale'] * 424
wcs_utils.set_axis_limits(
    center_ra=galaxy['ra'],
    center_dec=galaxy['dec'],
    arcsec_width=arcsec_width,
    wcs=target_image_wcs,
    ax=ax
)

# ax.imshow(desi_im, transform=ax.get_transform(wcs_desi))
# transform the segmap into the projection (desi) pixel space by referring to the segmap WCS
a = ax.imshow(mask, transform=ax.get_transform(segmap_wcs))
# .get_array() pulls masked array of pixels from imshow
# ma.getdata() converts to normal np array (not masked)
aligned_mask = np.ma.getdata(a.get_array())
"""


# def matplotlib_fig_to_numpy(fig):
#     # https://stackoverflow.com/questions/7821518/save-plot-to-numpy-array
#     io_buf = io.BytesIO()
#     fig.savefig(io_buf, format='raw', dpi=300)
#     io_buf.seek(0)
#     img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
#                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
#     io_buf.close()
#     return img_arr