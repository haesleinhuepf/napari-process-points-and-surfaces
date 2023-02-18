def test_create_surface():
    import napari_process_points_and_surfaces as nppas
    import numpy as np

    image = np.zeros((10, 10, 10))
    image[1:9, 1:9, 1:9] = 1
    image = image.astype(int)

    creator_functions = [
        nppas.marching_cubes,
        nppas.largest_label_to_surface,
        nppas.label_to_surface,
        nppas.all_labels_to_surface
    ]

    for func in creator_functions:

        surface = func(image)

        num_vertices = len(surface[0])
        num_faces = len(surface[1])

        assert num_vertices == 384
        assert num_faces == 764

def test_decimate():
    import napari_process_points_and_surfaces as nppas
    from functools import partial
    surface = nppas._vedo_stanford_bunny()

    num_vertices = len(surface[0])

    decimator_function = [
        partial(nppas.decimate_pro, fraction=0.1),
        partial(nppas.decimate_quadric, fraction=0.1),
        nppas.decimate_pro,
        nppas.decimate_quadric,
    ]

    for func in decimator_function:
        simplified_surface = func(surface)

        assert num_vertices > len(simplified_surface[0])


def test_create_convex_hull_from_surface():
    import napari_process_points_and_surfaces as nppas
    import numpy as np

    image = np.zeros((10, 10, 10))
    image[1:9, 1:9, 1:9] = 1
    image[:, 5:, 3:6] = 0

    surface = nppas.marching_cubes(image)

    convex_hull = nppas.create_convex_hull_from_surface(surface)

    assert len(convex_hull[0]) < len(surface[0])
    assert len(convex_hull[1]) < len(surface[1])

def test_remove_duplicate_vertices():
    import napari_process_points_and_surfaces as nppas
    gastruloid = nppas.gastruloid()

    # add a vertex
    a_list = list(gastruloid[0])
    a_list.append(gastruloid[0][0])
    gastruloid = (a_list, gastruloid[1])

    another_gastruloid = nppas.gastruloid()
    assert len(gastruloid[0]) > len(another_gastruloid[0])

    corrected_gastruloid = nppas.remove_duplicate_vertices(gastruloid)

    assert len(corrected_gastruloid[0]) <= len(another_gastruloid[0])

def test_surface_to_binary_volume():
    import napari_process_points_and_surfaces as nppas
    import numpy as np

    image = np.zeros((10, 10, 10))
    image[2:9, 2:9, 2:9] = 1
    image[:, 5:, 3:6] = 0

    surface = nppas.marching_cubes(image)
    binary = nppas.surface_to_binary_volume(surface, image)

    assert np.array_equal(binary, image)


def test_smooth_surface():

    import napari_process_points_and_surfaces as nppas
    gastruloid = nppas.gastruloid()

    smoothed_gastruloid = nppas.smooth_surface(gastruloid)

    # check if vertices/faces are still the same count
    assert len(gastruloid[0]) == len(smoothed_gastruloid[0])
    assert len(gastruloid[1]) == len(smoothed_gastruloid[1])


def test_subdivide():
    import napari_process_points_and_surfaces as nppas
    gastruloid = nppas.gastruloid()

    subdivision_functions = [
        nppas.subdivide_adaptive,
        # nppas.subdivide_loop_vedo, # see: https://github.com/haesleinhuepf/napari-process-points-and-surfaces/issues/new
        # nppas.subdivide_linear,
        # nppas.subdivide_butterfly
        ]

    for func in subdivision_functions:
        print(func)
        subdivided_gastruloid = func(gastruloid)

        # check if vertices/faces are still the same count
        assert len(gastruloid[0]) < len(subdivided_gastruloid[0])
        assert len(gastruloid[1]) < len(subdivided_gastruloid[1])

def test_sample_points_from_surface():
    import napari_process_points_and_surfaces as nppas
    gastruloid = nppas.gastruloid()

    points = nppas.sample_points_from_surface(gastruloid)
    assert len(points) == 1928


def test_subsample_points():
    import napari_process_points_and_surfaces as nppas
    gastruloid = nppas.gastruloid()

    points = nppas.sample_points_from_surface(gastruloid)

    subsampled_points = nppas.subsample_points(points, distance_fraction=0.1)

    assert len(points) > len(subsampled_points)

def test_create_convex_hull_from_points():
    import napari_process_points_and_surfaces as nppas
    gastruloid = nppas.gastruloid()

    points = nppas.sample_points_from_surface(gastruloid)

    convex_hull = nppas.create_convex_hull_from_points(points)
    assert len(convex_hull[0]) == 431
    assert len(convex_hull[1]) == 858
