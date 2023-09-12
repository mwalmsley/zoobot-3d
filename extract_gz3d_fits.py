# https://stackoverflow.com/questions/31028815/how-to-unzip-gz-file-using-python
import gzip
import shutil
import glob
from tqdm import tqdm

if __name__ == '__main__':

    # takes 20 mins on macbook air
    fits_gz_locs = list(glob.glob('data/gz3d/fits_gz/*.fits.gz'))
    for fits_gz_loc in tqdm(fits_gz_locs):
        # note the gzip.open
        with gzip.open(fits_gz_loc, 'rb') as f_in:
            with open(fits_gz_loc.replace('.gz', '').replace('fits_gz', 'fits'), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
