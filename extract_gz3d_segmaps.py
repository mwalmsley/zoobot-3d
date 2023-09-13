import json

from tqdm import tqdm
import pandas as pd
from astropy.io import fits


def save_segmaps(df: pd.DataFrame):

    for _, galaxy in tqdm(list(df.iterrows())):
        spiral_marks = extract_marks_for_galaxy(galaxy, 'spiral')
        bar_marks = extract_marks_for_galaxy(galaxy, 'bar')
        all_marks = {
            'spiral': spiral_marks,
            'bar': bar_marks
        }

        save_loc = galaxy['local_gz3d_fits_loc'].replace('/fits_gz/', '/segmaps/').replace('.fits.gz', '.json')
        # print(save_loc)
        with open(save_loc, 'w') as f:
            json.dump(all_marks, f)


def extract_marks_for_galaxy(galaxy, which_marks):

    if which_marks == 'spiral':
        fits_index = 9
    elif which_marks == 'bar':
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

    save_segmaps(pd.read_csv('/Users/user/repos/zoobot-3d/data/gz3d_and_gz_desi_master_catalog.csv'))

