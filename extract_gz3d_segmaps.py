import json
import os
import logging

import numpy as np
from tqdm import tqdm
import pandas as pd
from astropy.io import fits
from PIL import Image

from segmap_utils import construct_segmap_image



def save_segmaps(df: pd.DataFrame, overwrite=False):

    for _, galaxy in tqdm(list(df.iterrows())):
        spiral_marks = extract_marks_for_galaxy(galaxy, 'spiral_mask')
        bar_marks = extract_marks_for_galaxy(galaxy, 'bar_mask')
        all_marks = {
            'spiral_mask': spiral_marks,
            'bar_mask': bar_marks
        }
        
        marks_loc = galaxy['local_json_loc']
        if overwrite or not os.path.isfile(marks_loc):
            with open(marks_loc, 'w') as f:
                json.dump(all_marks, f)

        spiral_loc = galaxy['local_spiral_mask_loc']
        if overwrite or not os.path.isfile(spiral_loc):
            if len(spiral_marks) > 0:
                spiral_im = construct_segmap_image(galaxy, spiral_marks)
                if spiral_im.max() > 0:
                    Image.fromarray(spiral_im).save(spiral_loc)

        bar_loc = galaxy['local_bar_mask_loc']
        if overwrite or not os.path.isfile(bar_loc):
            if len(bar_marks) > 0:
                bar_im = construct_segmap_image(galaxy, bar_marks)
                if bar_im.max() > 0:
                    Image.fromarray(bar_im).save(bar_loc)


def extract_marks_for_galaxy(galaxy, which_marks):

    if which_marks == 'spiral_mask':
        fits_index = 9
    elif which_marks == 'bar_mask':
        fits_index = 10
    else:
        raise ValueError(which_marks)

    raw_annotations = fits.open(galaxy['local_gz3d_fits_loc'])[fits_index].data
    # third column, after classification_id and timestamp.
    # index of which user
    all_marks_by_users = [json.loads(r[2]) for r in raw_annotations]
    # list of N components, where each component is a list of XY pixel coordinate pairs constructing a path 

    # each user has an extra useless list level
    # all_marks_by_users = [x[0] for x in all_marks_by_users]

    return all_marks_by_users


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    
    



    if os.path.isdir('/share/nas2'):
        logging.info('Galahad paths')
        base_dir = '/share/nas2/walml/galaxy_zoo/segmentation/'
    else:
        logging.info('Local paths')
        base_dir = '/Users/user/repos/zoobot-3d/'

    df = pd.read_csv(base_dir + 'data/gz3d_and_gz_desi_master_catalog.csv')
    df = df.sample(len(df))  # for Galahad

    df['local_gz3d_fits_loc'] = base_dir + df['local_gz3d_fits_loc']
    df['local_json_loc'] = base_dir + df['local_json_loc']
    df['local_spiral_mask_loc'] = base_dir + df['local_spiral_mask_loc']
    df['local_bar_mask_loc'] = base_dir + df['local_bar_mask_loc']

    logging.info(df['local_gz3d_fits_loc'].iloc[0])
    
    logging.info(len(df))

    save_segmaps(df)
