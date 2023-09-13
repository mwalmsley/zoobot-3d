# https://github.com/tobiasgeron/download_DECaLS_images/blob/mike/downloader.py#L465
import numpy as np
from PIL import Image

def fits_to_png(im):
    # Set parameters for RGB image creation
    _scales = dict(
        g=(2, 0.008),
        r=(1, 0.014),
        z=(0, 0.019))
    _mnmx = (-0.5, 300)

    rgbimg = dr2_style_rgb(
        (im[0, :, :], im[1, :, :], im[2, :, :]),
        'grz',
        mnmx=_mnmx,
        arcsinh=1.,
        scales=_scales,
        desaturate=True)

    native_pil_image = Image.fromarray(
        np.uint8(rgbimg * 255.), mode='RGB')
    
    return native_pil_image


def dr2_style_rgb(imgs, bands, mnmx=None, arcsinh=None, scales=None, desaturate=False):
    '''
    Given a list of image arrays in the given bands, returns a scaled RGB image.
    Originally written by Dustin Lang and used by Kyle Willett for DECALS DR1/DR2 Galaxy Zoo subjects
    Args:
        imgs (list): numpy arrays, all the same size, in nanomaggies
        bands (list): strings, eg, ['g','r','z']
        mnmx (min,max), values that will become black/white *after* scaling. Default is (-3,10)):
        arcsinh (bool): if True, use nonlinear scaling (as in SDSS)
        scales (str): Override preset band scaling. Dict of form {band: (plane index, scale divider)}
        desaturate (bool): If [default=False] desaturate pixels dominated by a single colour
    Returns:
        (np.array) of shape (H, W, 3) with values between 0 and 1 of pixel values for colour image
    '''

    bands = ''.join(bands)  # stick list of bands into single string

    # first number is index of that band
    # second number is scale divisor - divide pixel values by scale divisor for rgb pixel value
    grzscales = dict(
        g=(2, 0.0066),
        r=(1, 0.01385),
        z=(0, 0.025),
    )

    if scales is None:
        if bands == 'grz':
            scales = grzscales
        elif bands == 'urz':
            scales = dict(
                u=(2, 0.0066),
                r=(1, 0.01),
                z=(0, 0.025),
            )
        elif bands == 'gri':
            scales = dict(
                g=(2, 0.002),
                r=(1, 0.004),
                i=(0, 0.005),
            )
        else:
            scales = grzscales

    #  create blank matrix to work with
    h, w = imgs[0].shape
    rgb = np.zeros((h, w, 3), np.float32)

    # Copy each band matrix into the rgb image, dividing by band scale divisor to increase pixel values
    for im, band in zip(imgs, bands):
        plane, scale = scales[band]
        rgb[:, :, plane] = (im / scale).astype(np.float32)

    # TODO mnmx -> (min, max)
    # cut-off values for non-linear arcsinh map
    if mnmx is None:
        mn, mx = -3, 10
    else:
        mn, mx = mnmx

    if arcsinh is not None:
        # image rescaled by single-pixel not image-pixel, which means colours depend on brightness
        rgb = nonlinear_map(rgb, arcsinh=arcsinh)
        mn = nonlinear_map(mn, arcsinh=arcsinh)
        mx = nonlinear_map(mx, arcsinh=arcsinh)

    # lastly, rescale image to be between min and max
    rgb = (rgb - mn) / (mx - mn)

    # default False, but downloader sets True
    if desaturate:
        # optionally desaturate pixels that are dominated by a single
        # colour to avoid colourful speckled sky

        # reshape rgb from (h, w, 3) to (3, h, w)
        RGBim = np.array([rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]])
        # a is mean pixel value across all bands, (h, w) shape
        a = RGBim.mean(axis=0)
        # putmask: given array and mask, set all mask=True values of array to new value
        # set pixels with 0 mean value to mean of 1. Inplace?
        np.putmask(a, a == 0.0, 1.0)
        # copy mean value array (h,w) into 3 bands (3, h, w)
        acube = np.resize(a, (3, h, w))
        # bcube: divide image by mean-across-bands pixel value, and again by 2.5 (why?)
        bcube = (RGBim / acube) / 2.5
        mask = np.array(bcube)  # isn't bcube already an array?
        # maximum per pixel across bands of mean-band-normalised rescaled image
        wt = np.max(mask, axis=0)
        # i.e largest relative deviation from mean
        # clip largest allowed relative deviation to one (inplace?)
        np.putmask(wt, wt > 1.0, 1.0)
        wt = 1 - wt  # invert relative deviations
        # non-linear rescaling of relative deviations
        wt = np.sin(wt*np.pi/2.0)
        # multiply by weights in complicated fashion
        temp = RGBim * wt + a*(1-wt) + a*(1-wt)**2 * RGBim
        rgb = np.zeros((h, w, 3), np.float32)  # reset rgb to be blank
        # fill rgb with weight-rescaled rgb
        for idx, im in enumerate((temp[0, :, :], temp[1, :, :], temp[2, :, :])):
            rgb[:, :, idx] = im

    clipped = np.clip(rgb, 0., 1.)  # set max/min to 0 and 1

    # logging.info(clipped)

    return clipped

def nonlinear_map(x, arcsinh=1.):
    """
    Apply non-linear map to input matrix. Useful to rescale telescope pixels for viewing.
    Args:
        x (np.array): array to have map applied
        arcsinh (np.float):
    Returns:
        (np.array) array with map applied
    """
    return np.arcsinh(x * arcsinh)
