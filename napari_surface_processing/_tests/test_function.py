# from napari_segment_blobs_and_things_with_membranes import threshold, image_arithmetic

# add your tests here...


def test_something():
    from napari_surface_processing import label_to_surface, largest_label_to_surface, convex_hull, \
        laplacian_smooth, taubin_smooth, simplification_clustering_decimation, colorize_curvature_apss

    import numpy as np

    from skimage.data import cells3d
    nuclei = cells3d()[:, 1, 60:120, 30:80]

    from skimage.measure import label
    labels = label(nuclei > 20000)

    surface = label_to_surface(labels, 3)

    surface = largest_label_to_surface(labels)

    convex_hull(surface)
    laplacian_smooth(surface)
    taubin_smooth(surface)
    simplification_clustering_decimation(surface)
    colorize_curvature_apss(surface)


