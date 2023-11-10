import glob

from astropy.io import fits
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':

    locs = list(glob.glob('data/gz3d/fits_gz/*.fits.gz'))
    # print(locs)
    data = []
    for loc in tqdm(locs):
        # print(loc)
        manga_metadata_for_galaxy = fits.open(loc)[5].data
        # np.recarray
        # print(manga_metadata_for_galaxy)
        columns = [x.name.lower() for x in manga_metadata_for_galaxy.columns]
        # print(columns)
        # print(dict(zip(columns, manga_metadata_for_galaxy[0])))
        # break
        data.append(dict(zip(columns, manga_metadata_for_galaxy[0])))

    df = pd.DataFrame(data=data)
    df['relative_gz3d_fits_loc'] = locs
    print(df.sample(5))
    df.to_csv('data/gz3d/reconstructed_gz3d_catalog_new.csv', index=False)
