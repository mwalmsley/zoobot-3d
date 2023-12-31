import sys
import os

from tqdm import tqdm
import pandas as pd




if __name__ == '__main__':

    # target_list_survey = 'manga'
    target_list_survey = 'sami'

    if os.path.isdir('/share/nas2'):
        galahad = True
        repo_dir = '/share/nas2/walml/repos/zoobot-3d'
        data_dir = '/share/nas2/walml/galaxy_zoo/segmentation/data'
        base_image_dir = os.path.join(data_dir, 'desi')
        
        sys.path.append('/share/nas2/walml/repos/download_DECaLS_images')
    else:
        galahad = False
        repo_dir = '/home/walml/repos/zoobot-3d'
        data_dir = os.path.join(repo_dir, 'data')
        # base_image_dir = os.path.join(data_dir, 'desi')
        # base_image_dir = '/Volumes/beta/galaxy_zoo/segmentation/data/desi'
        base_image_dir = os.path.join(data_dir, target_list_survey)      
        sys.path.append('/home/walml/repos/download_DECaLS_images')

    import downloader  # part of the private download_DECaLS_images repo above
    

    df = pd.read_csv(
        os.path.join(data_dir, f'{target_list_survey}_and_gz_desi_matches.csv')
    )

    # if not galahad:
    #     df = df[:100]
    
    for _, galaxy in tqdm(list(df.iterrows())):
        # center on MANGA source, not DESI source
        galaxy['ra'] = galaxy[f'ra_{target_list_survey}']
        galaxy['dec'] = galaxy[f'dec_{target_list_survey}']
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
            lazy_checking=False,
            print_url=False
        )
    