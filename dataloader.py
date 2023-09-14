import json
import logging
import io

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import segmap_utils
import wcs_utils

from galaxy_datasets.pytorch import galaxy_dataset

class SegmentationGalaxyDataset(galaxy_dataset.GalaxyDataset):

    def __init__(self, catalog: pd.DataFrame, label_cols=None, transform=None, target_transform=None, debug_alignment=False) -> None:
        catalog['segmap_json_loc'] = catalog['local_gz3d_fits_loc'].str.replace('/fits_gz/', '/segmaps/', regex=False).str.replace('.fits.gz', '.json', regex=False)
        # temp, until I update master catalog

        self.catalog = catalog
        self.label_cols = label_cols
        self.transform = transform
        self.target_transform = target_transform
        self.debug_alignment = debug_alignment


    def __getitem__(self, index: int):

        galaxy = self.catalog.iloc[index]

        # load the image into memory
        image_loc = galaxy['local_desi_jpg_loc']
        try:
            image = galaxy_dataset.load_img_file(image_loc)
            # HWC PIL image, 0-255 uint8
        except Exception as e:
            logging.critical('Cannot load {}'.format(image_loc))
            raise e

        # new bit - load segmaps
        with open(galaxy['segmap_json_loc'], 'r') as f:
            segmaps = json.load(f)
            segmap_dict = {}
            for segmap_name, marks_by_users in segmaps.items():
                segmap_image = construct_segmap_image(galaxy, marks_by_users, debug_alignment=self.debug_alignment)
                segmap_dict[segmap_name] = segmap_image
        
        if self.transform:
            image = self.transform(image)
            # apply to each segmap within the segmap_dict
            segmap_dict = dict([(name, self.transform(segmap)) for name, segmap in segmap_dict.copy().items()])

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



def construct_segmap_image(galaxy, marks_by_users, debug_alignment=False):

    # TODO could change the logic for iterating over users here
    mask = np.zeros((525, 525))
    for user_components in marks_by_users:
        mask += segmap_utils.draw_components(user_components, remove_self_intersecting=False)

    # work out the WCS of the segmap
    segmap_wcs = wcs_utils.get_wcs_assuming_image_size(
        center_ra=galaxy['ra_subject'],  # TODO
        center_dec=galaxy['dec_subject'],
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
   
    if debug_alignment:
        # debugging - make them on a matplotlib axi
        # for now, viz to check
        desi_im = Image.open(galaxy['local_desi_jpg_loc'])
        ax.imshow(desi_im, transform=ax.get_transform(target_image_wcs), alpha=.5)
        ax.axis('off')
        return aligned_mask, fig, ax
    else:
        plt.close()
        return aligned_mask


def matplotlib_fig_to_numpy(fig):
    # https://stackoverflow.com/questions/7821518/save-plot-to-numpy-array
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=300)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr


if __name__ == '__main__':

        
        catalog = pd.read_csv('/Users/user/repos/zoobot-3d/data/gz3d_and_gz_desi_master_catalog.csv')


        # minimal test of construct_segmap_image
        catalog['segmap_json_loc'] = catalog['local_gz3d_fits_loc'].str.replace('/fits_gz/', '/segmaps/', regex=False).str.replace('.fits.gz', '.json', regex=False)
        galaxy = catalog.iloc[32]
        with open(galaxy['segmap_json_loc'], 'r') as f:
            segmaps = json.load(f)
        spiral_marks = segmaps['spiral']
        _, fig, ax = construct_segmap_image(galaxy, spiral_marks, True)
        plt.savefig('temp_latest.png', bbox_inches='tight', pad_inches=0., facecolor='purple')

        # full test of dataloader
        # dataset = SegmentationGalaxyDataset(catalog)
        # image, segmap_dict = dataset[32]
        # fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(10, 3))
        # ax0.imshow(segmap_dict['spiral'])
        # ax1.imshow(segmap_dict['bar'])
        # ax2.imshow(image)
        # plt.show()


