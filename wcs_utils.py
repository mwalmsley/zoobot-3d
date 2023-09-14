
from astropy import wcs
import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u

def get_wcs_for_image(center_ra: float, center_dec: float, im: np.array, arcsec_per_pixel: float):
    # little wrapper for when we already know the image we're aiming for
    center_pixel_index = im.shape[1]/2
    return get_wcs_assuming_image_size(center_ra, center_dec, center_pixel_index, arcsec_per_pixel)


def get_wcs_assuming_image_size(center_ra, center_dec, center_pixel_index, arcsec_per_pixel):

    w = wcs.WCS(naxis=2)

    # example astropy use
    # https://docs.astropy.org/en/stable/wcs/example_create_imaging.html
    # explanation from
    # http://tdc-www.harvard.edu/wcstools/wcstools.wcs.html

    # CRPIX1 and CRPIX2 are the pixel coordinates of the reference point to which the projection and the rotation refer.
    # (technically, these would be the brick center coordinates I think, but hopefully it won't matter to approx as galaxy center) 
    w.wcs.crpix = [center_pixel_index, center_pixel_index]

    # CDELT1 and CDELT2 have been used to indicate the plate scale in degrees per pixel

    w.wcs.cdelt = np.array([arcsec_per_pixel/3600, arcsec_per_pixel/3600])

    # CRVAL1 and CRVAL2 give the center coordinate as right ascension and declination or longitude and latitude in decimal degrees. 
    w.wcs.crval = [center_ra, center_dec]
    # CTYPE1 and CTYPE2 indicate the coordinate type and projection. 
    # The first four characters are RA-- and DEC-, GLON and GLAT, or ELON and ELAT, for equatorial, galactic, and ecliptic coordinates, respectively.
    # The second four characters contain a four-character code for the projection. The presence of the CTYPE1 keyword is used to select for this WCS subset. 
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # https://fits.gsfc.nasa.gov/registry/tpvwcs/tpv.html
    # The TPV World Coordinate System is a non-standard convention following the rules of the WCS standard. 
    # It builds on the standard TAN projection by adding a general polynomial distortion correction
    # ...
    # Apply the distortion transformation using the coefficients in the PV keywords
    # (I don't need distortions here)
    # w.wcs.set_pv([(2, 1, 45.0)])

    return w


def set_axis_limits(center_ra, center_dec, arcsec_width, wcs, ax):

    # https://docs.sunpy.org/en/stable/generated/gallery/plotting/xy_lims.html
    xlims_world = center_ra * u.degree + np.array([-arcsec_width/2, arcsec_width/2]) * u.arcsec
    ylims_world = center_dec * u.degree + np.array([-arcsec_width/2, arcsec_width/2]) * u.arcsec

    world_coords = SkyCoord(ra=xlims_world, dec=ylims_world)
    pixel_coords_x, pixel_coords_y = wcs.world_to_pixel(world_coords)

    ax.set_xlim(pixel_coords_x)
    ax.set_ylim(pixel_coords_y)
    # acts inplace