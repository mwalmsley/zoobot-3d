import json
import os

import numpy as np
from tqdm import tqdm
import pandas as pd
from astropy.io import fits
from PIL import Image

from segmap_utils import construct_segmap_image



def save_segmaps(df: pd.DataFrame):

    for _, galaxy in tqdm(list(df.iterrows())):
        spiral_marks = extract_marks_for_galaxy(galaxy, 'spiral_mask')
        bar_marks = extract_marks_for_galaxy(galaxy, 'bar_mask')
        all_marks = {
            'spiral_mask': spiral_marks,
            'bar_mask': bar_marks
        }

        with open(galaxy['local_json_loc'], 'w') as f:
            json.dump(all_marks, f)

        if len(spiral_marks) > 0:
            spiral_im = construct_segmap_image(galaxy, spiral_marks)
            Image.fromarray(spiral_im).save(galaxy['local_spiral_mask_loc'])
        if len(bar_marks) > 0:
            bar_im = construct_segmap_image(galaxy, bar_marks)
            Image.fromarray(bar_im).save(galaxy['local_bar_mask_loc'])


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


# def get_mask_image_for_galaxy(galaxy, user_marks):
#     mask = construct_segmap_image(galaxy, user_marks)
#     # print(mask.max())
#     # exit()
#     # mask = (mask*255/mask.max()).astype(np.uint8)
#     # print(mask.min(), mask.max())
#     return 

            

if __name__ == '__main__':

    df = pd.read_csv('/Users/user/repos/zoobot-3d/data/gz3d_and_gz_desi_master_catalog.csv')

    print(len(df))

    save_segmaps(df)

