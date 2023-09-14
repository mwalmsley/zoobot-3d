import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from shapely.geometry import LineString



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
