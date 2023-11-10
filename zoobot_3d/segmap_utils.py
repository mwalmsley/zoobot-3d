import logging

# import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from shapely.geometry import LineString
from PIL import Image



def construct_segmap_image(galaxy, marks_by_users):

    manga_segmap_dim = 525

    # TODO could change the logic for iterating over users here
    mask = np.zeros((manga_segmap_dim, manga_segmap_dim))
    for user_components in marks_by_users:
        mask += draw_components(user_components, remove_self_intersecting=False)

    # mask needs to be flipped by convention
    mask = mask[::-1]

    # convert to RGB image
    if mask.max() == 0:  # empty mask :(
        mask_im = Image.fromarray(np.zeros((manga_segmap_dim, manga_segmap_dim)).astype(np.uint8))
    else:
        # mask_im = Image.fromarray((255*mask/mask.max()).astype(np.uint8))
        # don't just normalise by mask.max(), because this is affected by num. intersections
        # instead, normalise by fraction of volunteers who made marks
        """
        num marking / num asked to mark (always 15)
        """
        num_users = len(marks_by_users)
        if not num_users == 15:
            logging.warning('Skipping as {} marks: {}'.format(num_users, galaxy['segmap_json_loc']))
        else:
            # mask is initially from 0 to AT MOST 15, divide by 15 to get to 0 to 1
            mask_im = Image.fromarray((15*mask/num_users).astype(np.uint8))  # const norm
            # mask_im = Image.fromarray((10*mask/num_users).astype(np.uint8))  # const norm

    # align to DESI FoV

    # mask will be centered at same location, but different FoV (52'') and pixscale (0.099''/pixel)
    desi_field_of_view = galaxy['est_dr5_pixscale'] * 424
    segmap_pixels_needed = desi_field_of_view / 0.099  # height/width required for desi FoV
    extra_pixels_needed = (segmap_pixels_needed - manga_segmap_dim) // 2

    # negative crop on all sides of manga image, to extend as needed
    # 0-padded by default
    
    left = - extra_pixels_needed
    upper = - extra_pixels_needed
    right = manga_segmap_dim + extra_pixels_needed
    lower = manga_segmap_dim + extra_pixels_needed
    mask_im = mask_im.crop((left, upper, right, lower))

    # resize to DESI jpg image size
    mask_im = mask_im.resize((424, 424))

    return np.array(mask_im)



def draw_components(marks_of_components, remove_self_intersecting=False) -> np.ndarray:
    # list of components, with each component being a list of XY pixel pairs
    # may want to wrap this in a for loop, if iterating over users

    dimensions = (525, 525)  # probably cannot be changed, else the marks won't make sense
    coords = [[x, y] for y in range(dimensions[1]) for x in range(dimensions[0])]

    mask = np.zeros(dimensions)
    for marked_component in marks_of_components:
            mask += draw_component(marked_component, dimensions=dimensions, coords=coords, remove_self_intersecting=remove_self_intersecting)

    return mask[::-1]  # always flip, by convention so up in y direction is increase in dec

# https://github.com/CKrawczyk/GZ3D_production/blob/master/make_subject_fits.py#L140C17-L140C17
# refactored to act on single component not for component-by-user
def draw_component(X, dimensions, coords, remove_self_intersecting):
    # X is vector marks, single list of XY pairs
    codes = [Path.MOVETO] + [Path.LINETO] * len(X)

    if len(X) > 1:
        if remove_self_intersecting and not LineString(X).is_simple:
            return np.zeros(dimensions)
            #     # remove self intersecting paths
            # "any self-intersections are only at boundary points"
    p_closed = X + [X[0]]
    mpl_path = Path(p_closed, codes=codes)
    # this highlights the 'inside'
    # assumes path is closed - in practice, connects last vertex back to first vertex
    inside_component = mpl_path.contains_points(coords).reshape(*dimensions)

    return inside_component
