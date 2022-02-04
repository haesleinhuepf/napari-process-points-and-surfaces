
__version__ = "0.1.0"
__common_alias__ = "nsurfpro"

from napari.types import SurfaceData
from napari.types import ImageData, LabelsData

from napari_plugin_engine import napari_hook_implementation
from napari_tools_menu import register_function
import numpy as np
from enum import Enum

from scipy import ndimage as ndi

from skimage import filters
import scipy
from scipy import ndimage
import napari

from napari_time_slicer import time_slicer

@napari_hook_implementation
def napari_experimental_provide_function():
    return [label_to_surface, convex_hull, laplacian_smooth, taubin_smooth, simplification_clustering_decimation, colorize_curvature_apss
    ]

@register_function(menu="Surfaces > Any label to surface (marching cubes, scikit-image, nsurfpro)")
@time_slicer
def label_to_surface(labels: LabelsData, label_id: int = 1) -> SurfaceData:
    """
    Turn a single label out of a label image into a surface using the marching cubes algorithm
    """
    from skimage.measure import marching_cubes

    binary = np.asarray(labels == label_id)

    vertices, faces, normals, values = marching_cubes(binary, 0)

    return (vertices, faces, values)


@register_function(menu="Surfaces > Largest label to surface (marching cubes, scikit-image, nsurfpro)")
@time_slicer
def largest_label_to_surface(labels: LabelsData, label_id: int = 1) -> SurfaceData:
    """
    Turn a the largest label in of a label image into a surface using the marching cubes algorithm
    """
    from skimage.measure import regionprops
    statistics = regionprops(labels)

    label_index = np.argmax([r.area for r in statistics])
    labels_list = [r.label for r in statistics]
    label = labels_list[label_index]

    return label_to_surface(labels, label)


@register_function(menu="Surfaces > Convex hull (pymeshlab, nsurfpro)")
def convex_hull(surface: SurfaceData, viewer:napari.Viewer=None) -> SurfaceData:
    import pymeshlab
    mesh = pymeshlab.Mesh(surface[0], surface[1])
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh)
    ms.set_current_mesh(0)

    ms.convex_hull()

    mesh = ms.mesh(1)

    faces = np.asarray(mesh.polygonal_face_list())
    vertices = np.asarray(mesh.vertex_matrix())
    values = np.asarray(mesh.vertex_color_array())

    return (vertices, faces, values)


@register_function(menu="Surfaces > Laplacian smooth (pymeshlab, nsurfpro)")
def laplacian_smooth(surface: SurfaceData, steps_mooth_num: int = 10,
                            viewer:napari.Viewer=None) -> SurfaceData:
    import pymeshlab
    mesh = pymeshlab.Mesh(surface[0], surface[1])
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh)
    ms.set_current_mesh(0)
    ms.laplacian_smooth(stepsmoothnum=steps_mooth_num)
    mesh = ms.mesh(0)

    faces = np.asarray(mesh.polygonal_face_list())
    vertices = np.asarray(mesh.vertex_matrix())
    values = np.ones((len(vertices)))

    return (vertices, faces, values)

@register_function(menu="Surfaces > Taubin smooth (pymeshlab, nsurfpro)")
def taubin_smooth(surface: SurfaceData,
                  lambda_: float = 0.5,
                  mu: float = -0.53,
                  step_smooth_num: int = 10,
                  viewer:napari.Viewer=None
                  ) -> SurfaceData:
    import pymeshlab

    mesh = pymeshlab.Mesh(surface[0], surface[1])
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh)
    ms.set_current_mesh(0)
    ms.taubin_smooth(lambda_=lambda_,
                     mu=mu,
                     stepsmoothnum=step_smooth_num
                     )

    mesh = ms.mesh(0)

    faces = np.asarray(mesh.polygonal_face_list())
    vertices = np.asarray(mesh.vertex_matrix())
    values = np.ones((len(vertices)))

    return (vertices, faces, values)


@register_function(menu="Surfaces > Simplification clustering decimation (pymeshlab, nsurfpro)")
def simplification_clustering_decimation(surface: SurfaceData,
                                         threshold_percentage: float = 1,
                                         viewer:napari.Viewer=None
                                         ) -> SurfaceData:
    import pymeshlab

    mesh = pymeshlab.Mesh(surface[0], surface[1])
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh)
    ms.set_current_mesh(0)
    ms.simplification_clustering_decimation(threshold=pymeshlab.Percentage(threshold_percentage))
    mesh = ms.mesh(0)

    faces = np.asarray(mesh.polygonal_face_list())
    vertices = np.asarray(mesh.vertex_matrix())
    values = np.ones((len(vertices)))

    return (vertices, faces, values)




class CurvatureType(Enum):
    mean = 'Mean'
    gauss = 'Gauss'
    k1 = 'K1'
    k2 = 'K2'
    approxmean = 'ApproxMean'

@register_function(menu="Surfaces > Colorize curvature (apss, pymeshlab, nsurfpro)")
def colorize_curvature_apss(surface: SurfaceData,
                            filter_scale: float = 2,
                            projection_accuracy: float = 0.0001,
                            max_projection_iterations: int = 15,
                            spherical_parameter: float = 1,
                            curvature_type: CurvatureType = CurvatureType.mean,
                            viewer:napari.Viewer=None
                            ) -> SurfaceData:

    import pymeshlab

    mesh = pymeshlab.Mesh(surface[0], surface[1])
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh)
    ms.set_current_mesh(0)
    ms.colorize_curvature_apss(
        filterscale=filter_scale,
        projectionaccuracy=projection_accuracy,
        maxprojectioniters=max_projection_iterations,
        sphericalparameter=spherical_parameter,
        curvaturetype=curvature_type.value
    )

    mesh = ms.mesh(0)

    faces = np.asarray(mesh.polygonal_face_list())
    vertices = np.asarray(mesh.vertex_matrix())
    values = np.asarray(mesh.vertex_color_array())

    return (vertices, faces, values)

