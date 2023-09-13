# zoobot-3d

Segmentation playground for GZ3D and more.

## Plan

The general plan is:

* Use WCS to line up the GZ3D masks with the DESI-LS cutout service fits for MANGA (in progress)
* Subclass the galaxy-datasets dataloader to add support for segmentation masks
* Model surgery: separate out the encoder and decoder halves and define as separate forward pass steps. Add classification head (zoobot style, Dirichlet loss) to optionally forward pass instead of the decoder half. (in progress)
* Check the UNet encoder + decoder works for segmentation
* Check the UNet encoder + classifier head works for classifications
* Jointly learn to segment and classify, by doing a forward pass through both the decoder and classifier (separately) and toggling the appropriate loss per image and accumulating the gradients over a mixed batch. Pray it does better than the UNet baseline. Alternatively we could use a pretrained encoder (Zoobot) but I think this would work better and be no harder to implement.
* Measure performance vs segmentation-only baseline
* Make predictions on DESI-LS images of SAMI galaxies (90% are within DESI-LS)

## Data Prep.

GZ3D is available from Marvin, which is a PITA to install, so instead we scrape the server for the GZ3D FITS. `download_gz3d_fits.py`.
Each FITS includes: segmentation maps for spirals and bars, raw vector paths from volunteers for spirals and bars, and MANGA metadata.

We construct a catalog of GZ3D galaxies using the MANGA metadata in the GZ3D fits: `construct_gz3d_catalog.py`.

We cross-match that catalog (plus the SAMI catalog, see below) with the GZ DESI base catalog: `desi_manga_crossmatch.ipynb`.

We add the GZ DESI fits file locs to the catalog, and grab them from Manchester with rsync: `grab_desi_fits.ipynb`. The catalog is now ready: `data/gz3d_and_gz_desi_master_catalog.csv`

We already have DESI images resized at 424x424 and cropped to a GZ DESI FoV.

For training, we will use only galaxies which have spiral and/or bar segmaps.

## Data Loaders

For each galaxy, we dynamically construct GZ3D segmaps from the raw volunteer classifications. We use WCS to place these in an image with pixel positions that match the corresponding GZ DESI image. We can choose which volunteer classifications to use.


---


For much later, we have downloaded the SAMI candidate target list.

* Fixed pixel scale.
* [Latest data release paper](https://academic.oup.com/mnras/article/505/1/991/6123881) (main difference is +800 cluster galaxies)
* [SAMI DR3 overview](http://www.sami-survey.org/node/902)
* [Great docs](https://docs.datacentral.org.au/sami/data-release-3/)
* [Explaining the IFU data products](https://docs.datacentral.org.au/sami/data-release-3/core-data-products/)
* [Index of access services](https://datacentral.org.au/services/)
* Target selection is from GAMA. Here’s the [docs](https://docs.datacentral.org.au/sami/data-release-3/input-and-photometric-catalogues/).
