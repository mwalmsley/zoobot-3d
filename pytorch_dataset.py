import json
import logging
import io

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
        with open(galaxy['segmap_json_loc'], 'r') as f:
            segmaps = json.load(f)
            segmap_dict = {}
            for segmap_name, marks_by_users in segmaps.items():
                segmap_image = construct_segmap_image(galaxy, marks_by_users)
                segmap_dict[segmap_name] = segmap_image
        
        if self.transform:
            transformed = self.transform(image=image, **segmap_dict)
            image = transformed['image']
            # everything else is the transformed segmap dict
            del transformed['image']
            segmap_dict = transformed
            # segmap_dict = dict([(k, v) for k, v in transformed.items() if k != 'image'])

        if self.label_cols is None:
            return image, segmap_dict
        else:
            # load the labels. If no self.label_cols, will 
            label = galaxy_dataset.get_galaxy_label(galaxy, self.label_cols)    

            # I never use this tbh
            if self.target_transform:
                label = self.target_transform(label)
            # no effect on mask labels

            return image, segmap_dict, label



def construct_segmap_image(galaxy, marks_by_users):

    manga_segmap_starting_dim = 525

    # TODO could change the logic for iterating over users here
    mask = np.zeros((manga_segmap_starting_dim, manga_segmap_starting_dim))
    for user_components in marks_by_users:
        mask += segmap_utils.draw_components(user_components, remove_self_intersecting=False)

    # mask needs to be flipped by convention
    mask = mask[::-1]

    # convert to RGB image
    assert mask.max() > 0
    assert mask.min() == 0
    mask_im = Image.fromarray((255*mask/mask.max()).astype(np.uint8))

    # align to DESI FoV

    # mask will be centered at same location, but different FoV (52'') and pixscale (0.099''/pixel)
    desi_field_of_view = galaxy['est_dr5_pixscale'] * 424
    segmap_pixels_needed = desi_field_of_view / 0.099  # height/width required for desi FoV
    extra_pixels_needed = (segmap_pixels_needed - manga_segmap_starting_dim) // 2

    # negative crop on all sides of manga image, to extend as needed
    # 0-padded by default
    
    left = - extra_pixels_needed
    upper = - extra_pixels_needed
    right = manga_segmap_starting_dim + extra_pixels_needed
    lower = manga_segmap_starting_dim + extra_pixels_needed
    mask_im = mask_im.crop((left, upper, right, lower))

    # resize to DESI jpg image size
    mask_im = mask_im.resize((424, 424))

    return np.array(mask_im)



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
        image, segmap_dict = dataset[32]

        fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(10, 3))
        ax0.imshow(segmap_dict['spiral_mask'])
        ax1.imshow(segmap_dict['bar_mask'])
        ax2.imshow(image)  # default target keyed as image
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