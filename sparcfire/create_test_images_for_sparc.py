import os
import logging

import cv2
import pandas as pd


def main():

    on_local = os.path.isdir('/Users/user/')
    if on_local:
        base_dir = '/Users/user/repos/zoobot-3d/'
    else:
        base_dir = '/share/nas2/walml/galaxy_zoo/segmentation/'

    df = pd.read_parquet(base_dir + 'data/test_catalog.parquet')
    logging.info(df['spiral_mask_loc'].iloc[0])
    df['spiral_mask_exists'] = df['spiral_mask_loc'].apply(os.path.isfile)
    assert any(df['spiral_mask_exists'])
    df = df.query('spiral_mask_exists').reset_index(drop=True)

    df = df[:256]

    for _, galaxy in df.iterrows():
        create_cropped_image(galaxy['desi_jpg_loc'], base_dir + 'data/sparcfire/test_images/')


def create_cropped_image(image_loc, save_dir, h=256, w=256, overwrite=False):

    save_loc = save_dir + image_loc.split('/')[-1]

    if os.path.isfile(save_loc) and not overwrite:
        return

    img = cv2.imread(image_loc)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    center = img.shape
    x = center[1]/2 - w/2
    y = center[0]/2 - h/2

    crop_img = img[int(y):int(y+h), int(x):int(x+w)]
    cv2.imwrite(save_loc, crop_img)


if __name__ == '__main__':
    main()
