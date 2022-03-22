# from napari_segment_blobs_and_things_with_membranes import threshold, image_arithmetic

# add your tests here...


def test_something(make_napari_viewer):
    from napari_process_points_and_surfaces import convex_hull,\
            filter_smooth_simple,\
            filter_smooth_laplacian,\
            filter_smooth_taubin,\
            simplify_vertex_clustering,\
            simplify_quadric_decimation,\
            subdivide_loop,\
            labels_to_centroids,\
            sample_points_uniformly,\
            sample_points_poisson_disk,\
            voxel_down_sample,\
            points_to_labels,\
            points_to_convex_hull_surface,\
            surface_from_point_cloud_alpha_shape,\
            surface_from_point_cloud_ball_pivoting,\
            label_to_surface,\
            largest_label_to_surface,\
            add_quality,\
            Quality

    viewer = make_napari_viewer()

    from skimage.data import cells3d
    nuclei = cells3d()[:, 1, 60:120, 30:80]

    from skimage.measure import label
    labels = label(nuclei > 20000)
    viewer.add_labels(labels)

    surface = label_to_surface(viewer.layers[0], 3)

    surface = largest_label_to_surface(viewer.layers[0])

    convex_hull(surface)
    filter_smooth_simple(surface)
    filter_smooth_taubin(surface)
    filter_smooth_laplacian(surface)
    simplify_quadric_decimation(surface)
    simplify_vertex_clustering(surface)
    subdivide_loop(surface)
    labels_to_centroids(labels)
    sample_points_uniformly(surface)
    points = sample_points_poisson_disk(surface)
    voxel_down_sample(points)
    points_to_labels(points, labels)
    points_to_convex_hull_surface(points)
    surface_from_point_cloud_ball_pivoting(points)
    surface_from_point_cloud_alpha_shape(points)
    add_quality(surface, Quality.SKEW)

def test_curvature():
    from napari_process_points_and_surfaces import add_curvature_scalars,\
        Curvature,\
        add_spherefitted_curvature
    import vedo
    import numpy as np

    shape = vedo.shapes.Ellipsoid()
    surface_data = (shape.points(), np.asarray(shape.faces()))

    add_curvature_scalars(surface_data, Curvature.Gauss_Curvature)
    add_curvature_scalars(surface_data, Curvature.Mean_Curvature)
    add_curvature_scalars(surface_data, Curvature.Maximum_Curvature)
    add_curvature_scalars(surface_data, Curvature.Minimum_Curvature)

    add_spherefitted_curvature(surface_data, radius=1)
