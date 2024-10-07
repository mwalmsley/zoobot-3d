# zoobot-3d (NeurIPS 2023 version)

Segmentation playground for GZ3D and more.

## Plan

The general plan is:

* Line up the GZ3D masks with the DESI-LS cutout service fits for MANGA (done)
* Subclass the galaxy-datasets dataloader to add support for segmentation masks (done)
* Model surgery: separate out the encoder and decoder halves and define as separate forward pass steps. Add classification head (zoobot style, Dirichlet loss) to optionally forward pass instead of the decoder half. (in progress)
* Check the UNet encoder + decoder works for segmentation
* Check the UNet encoder + classifier head works for classifications
* Jointly learn to segment and classify, by doing a forward pass through both the decoder and classifier (separately) and toggling the appropriate loss per image and accumulating the gradients over a mixed batch. Pray it does better than the UNet baseline. Alternatively we could use a pretrained encoder (Zoobot) but I think this would work better and be no harder to implement.
* Measure performance vs segmentation-only baseline
* Make predictions on DESI-LS images of SAMI galaxies (90% are within DESI-LS)

## Input Data

### MANGA Notes

GZ3D is available from Marvin, which is a PITA to install, so instead we scrape the server for the GZ3D FITS. `download_gz3d_fits.py`.
Each FITS includes: segmentation maps for spirals and bars, raw vector paths from volunteers for spirals and bars, and MANGA metadata.

### SAMI Notes

Downloaded the SAMI candidate target list.

* Fixed pixel scale.
* [Latest data release paper](https://academic.oup.com/mnras/article/505/1/991/6123881) (main difference is +800 cluster galaxies)
* [SAMI DR3 overview](http://www.sami-survey.org/node/902)
* [Great docs](https://docs.datacentral.org.au/sami/data-release-3/)
* [Explaining the IFU data products](https://docs.datacentral.org.au/sami/data-release-3/core-data-products/)
* [Index of access services](https://datacentral.org.au/services/)
* Target selection is from GAMA. Here’s the [docs](https://docs.datacentral.org.au/sami/data-release-3/input-and-photometric-catalogues/).

### Other Data

We also use the GZ DESI volunteer votes and ML predictions.

## Prepare Data

### Turn the Volunteer Annotations into Images

Extract the individual volunteer marks (vector vertices) and save them as JSON text to `segmap_json_loc`. We no longer need the GZ3D FITS themselves.

Load these JSON to construct jpg "images" where the pixel value is proportional to the fraction of volunteers enclosing that pixel. `extract_gz3d_segmaps.py`, with image construction imported from `zoobot_3d.segmap_utils.construct_segmap_image`. Images saved to `spiral_mask_loc` and `bar_mask_loc`. Skips if no marks.

### Cross-match to DESI and Download New DESI Images

Construct a catalog of GZ3D galaxies using the MANGA metadata in the GZ3D fits: `construct_gz3d_catalog.py`, creating `reconstructed_gz3d_catalog_new.csv` (which is really just a list of the GZ3D segmap files)

Cross-match the core GZ DESI catalog (`data/desi/master_all_file_index_passes_file_checks.parquet`) with the three catalogs we're interested in here: GZ3D/MANGA, SAMI, and GZ2. `crossmatch_catalogs.ipynb`.

I already have images for all these galaxies, but, they are all centered on the DESI catalog source coordinates rather than the MANGA catalog source coordinates. It's easier to redownload than to mess around with WCS.
<!-- , and grab them from Manchester with rsync: `grab_desi_fits.ipynb`.  -->
`download_centered_desi_cutouts` downloads *new* DESI FITS and jpg images, centered precisely on the manga segmaps (but otherwise identical to the GZ DESI images).

`final_catalog_tweaks.ipynb` tinkers with paths and merges in the GZ DESI and GZ2 vote tables.
There's now quite a lot of columns, so to summarise, `data/gz3d_and_desi_master_catalog.parquet` has:

* GZ3D FITS metadata (RA, Dec, gz_spiral_votes, etc). Not used.
* DESI master catalog columns inc. dr8_id, est_dr5_pixscale. Crucial!
* Paths to the images/segmaps. `relative_desi_jpg_loc`, `relative_segmap_json_loc`, `relative_spiral_mask_loc`, `relative_bar_mask_loc`
* GZ DESI ML predicted vote fractions for *these four columns only*, for filtering:  `smooth-or-featured_featured-or-disk_fraction`, `disk-edge-on_yes_fraction`, `has-spiral-arms_yes_fraction`, `spiral-arm-count_2_fraction`
* GZ DESI volunteer votes e.g. `smooth-or-featured-dr8_smooth_fraction`
* GZ2 volunteer votes e.g. `smooth-or-featured-gz2_smooth_fraction`

## Deep Learning



### Data Loading

I tried constructing the label segmaps 'on the fly', but it's much too slow for practical training.

`zoobot_3d.pytorch_dataset.py` has a Dataset class which, given a dataframe of galaxies:

* loads the centered image (from `desi_jpg_loc`)
* optionally, loads the GZ3D spiral/bar masks (`spiral_mask_loc`, `bar_mask_loc`)
* optionally, loads the vote labels
* optionally, applies an albumentations transform to the image and masks (consistently)

The Dataset yields dicts like

    {
        'image': (augmented DESI RGB image, 0-1 floats),
        'spiral_mask': (augmented mask from GZ3D, 0-255 uint where value is prop. to volunteers enclosing that pixel),
        'bar_mask': (similarly),
        'label_cols': classification votes as usual with Zoobot e.g. [[1, 4, 2, ...], ...]. No longer used.
    }

`zoobot_3d.pytorch_datamodule.py` has LightningDataModule class which creates train, val and test Datasets from `zoobot_3d.pytorch_dataset.py`.

For training, we currently use only galaxies which have spiral and/or bar segmaps.

### Model

It's a UNet adapted from Mike Smith's diffusion paper.

I did an hparam sweep of the key bits; see slurm/sweep.yaml.

The head predicts the fraction of volunteers enclosing each pixel. The loss function is calculated on this vs the actual fraction (as encoded on the segmap jpgs). The loss func. can be L1, MSE, or Beta-Binomial; we found L1 worked best visually ('softer' than MSE, and BB didn't train well).

### Training

Nothing unusual, see `train.py`. Importantly, we only train on galaxies with spiral volunteer segmaps.

(this was more complicated when training on segmaps+votes, but we removed that)

---

## Results for Paper

`predict.py` loops over a catalog of galaxies. Used for all MANGA galaxies and, soon, all SAMI galaxies. 

`create_comparison_grids.py` makes paper-ready figures for comparing GZ3D, sparcfire, and our model.



## Useful commands

    rsync --files-from data/galahad_jpg_to_copy.txt -e 'ssh -A -J walml@external.jb.man.ac.uk' walml@galahad.ast.man.ac.uk:/share/nas2/walml/galaxy_zoo/decals/dr8/jpg data/gz_desi/jpg

    rsync -az -e 'ssh -A -J walml@external.jb.man.ac.uk' data walml@galahad.ast.man.ac.uk:/share/nas2/walml/galaxy_zoo/segmentation

    rsync -avz -e 'ssh -A -J walml@external.jb.man.ac.uk' walml@galahad.ast.man.ac.uk:/share/nas2/walml/galaxy_zoo/segmentation/data/desi/jpg data/desi

    rsync -avz -e 'ssh -A -J walml@external.jb.man.ac.uk' /Users/user/repos/zoobot-3d/data/*.parquet walml@galahad.ast.man.ac.uk:/share/nas2/walml/galaxy_zoo/segmentation/data

    rsync -avz -e 'ssh -A -J walml@external.jb.man.ac.uk' walml@galahad.ast.man.ac.uk:/share/nas2/walml/galaxy_zoo/segmentation/data/gz3d/segmaps/masks data/gz3d/segmaps


    <!-- current best sweep model -->
    rsync -avz -e 'ssh -A -J walml@external.jb.man.ac.uk' walml@galahad.ast.man.ac.uk:/share/nas2/walml/galaxy_zoo/segmentation/outputs/run_1695899881.3925836 outputs

    <!-- same, but slightly zoomed images -->
    rsync -avz -e 'ssh -A -J walml@external.jb.man.ac.uk' walml@galahad.ast.man.ac.uk:/share/nas2/walml/galaxy_zoo/segmentation/outputs/run_1695938854.2480044  outputs
