import sys
import os

from tqdm import tqdm
import pandas as pd




if __name__ == '__main__':

    if os.path.isdir('/share/nas2'):
        galahad = True
        repo_dir = '/share/nas2/walml/repos/zoobot-3d'
        data_dir = '/share/nas2/walml/galaxy_zoo/segmentation/data'
        base_image_dir = os.path.join(data_dir, 'desi')
        
        sys.path.append('/share/nas2/walml/repos/download_DECaLS_images')
    else:
        galahad = False
        repo_dir = '/Users/user/repos/zoobot-3d'
        data_dir = os.path.join(repo_dir, 'data')
        # base_image_dir = os.path.join(data_dir, 'desi')
        base_image_dir = '/Volumes/beta/galaxy_zoo/segmentation/data/desi'
        sys.path.append('/Users/user/repos/download_DECaLS_images')

    import downloader
    

    df = pd.read_csv(
        os.path.join(
            data_dir, 'gz3d_and_gz_desi_matches.csv')
    )

    if not galahad:
        df = df[:100]
    
    for _, galaxy in tqdm(list(df.iterrows())):
        # center on MANGA source, not DESI source
        galaxy['ra'] = galaxy['ra_manga']
        galaxy['dec'] = galaxy['dec_manga']
        rgb_format = 'jpg'

        fits_dir = os.path.join(base_image_dir, 'fits')
        rgb_dir = os.path.join(base_image_dir, rgb_format)

        for temp_dir in [base_image_dir, fits_dir, rgb_dir]:
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

        downloader.download_images(
            galaxy,
            base_image_dir,
            rgb_format='jpg', # originally png
            data_release='8',
            force_redownload=False,
            force_new_rgb=False,
            max_retries=5,
            min_pixelscale=0.1,
            rgb_size=424,
            lazy_checking=True,
            print_url=False
        )
    