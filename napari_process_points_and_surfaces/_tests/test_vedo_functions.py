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
