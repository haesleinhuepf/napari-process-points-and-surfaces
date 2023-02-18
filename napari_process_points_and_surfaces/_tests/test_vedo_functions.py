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



