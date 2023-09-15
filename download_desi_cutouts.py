import sys
sys.path.append('/Users/user/repos/download_DECaLS_images')
import os

from tqdm import tqdm
import pandas as pd

import downloader


if __name__ == '__main__':

    df = pd.read_csv(
        '/Users/user/repos/zoobot-3d/data/gz3d_and_gz_desi_matches.csv'
    )[:100]
    # galaxy = df.iloc[32]
    for _, galaxy in tqdm(list(df.iterrows())):
        # center on MANGA source, not DESI source
        galaxy['ra'] = galaxy['ra_manga']
        galaxy['dec'] = galaxy['dec_manga']
        base_dir = 'data/desi'
        rgb_format = 'jpg'

        fits_dir = os.path.join(base_dir, 'fits')
        rgb_dir = os.path.join(base_dir, rgb_format)

        for temp_dir in [base_dir, fits_dir, rgb_dir]:
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

        downloader.download_images(
            galaxy,
            base_dir,
            rgb_format='jpg', # originally png
            data_release='8',
            force_redownload=False,
            force_new_rgb=False,
            max_retries=5,
            min_pixelscale=0.1,
            rgb_size=424,
            lazy_checking=False,
            print_url=False
        )
    