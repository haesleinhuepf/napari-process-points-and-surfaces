
__version__ = "0.4.1"
__common_alias__ = "nppas"

import warnings


from napari_plugin_engine import napari_hook_implementation
from napari_tools_menu import register_function, register_action
import numpy as np
import stackview

from ._surface_annotation_widget import SurfaceAnnotationWidget

from napari_time_slicer import time_slicer
from ._quantification import add_quality, Quality, add_curvature_scalars,\
    Curvature, add_spherefitted_curvature, surface_quality_table, \
    surface_quality_to_properties, set_vertex_values

from ._vedo import (to_vedo_mesh,
                    to_vedo_points,
                    to_napari_surface_data,
                    to_napari_points_data,
                    smooth_surface,
                    _subdivide_loop_vedo,
                    _subdivide_linear,
                    subdivide_adaptive,
                    _subdivide_butterfly,
                    subdivide_centroid,
                    sample_points_from_surface,
                    subsample_points,
                    create_convex_hull_from_points,
                    create_convex_hull_from_surface,
                    fill_holes_in_surface,
                    remove_duplicate_vertices,
                    decimate_quadric,
                    decimate_pro,
                    show,
                    SurfaceTuple,
                    smooth_surface_moving_least_squares_2d,
                    smooth_surface_moving_least_squares_2d_radius,
                    smooth_pointcloud_moving_least_squares_2d_radius,
                    smooth_pointcloud_moving_least_squares_2d,
                    reconstruct_surface_from_pointcloud
                    )

from ._utils import isotropic_scale_surface


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return SurfaceAnnotationWidget


@napari_hook_implementation
def napari_experimental_provide_function():
    return [convex_hull,
            filter_smooth_simple,
            filter_smooth_laplacian,
            filter_smooth_taubin,
            simplify_vertex_clustering,
            simplify_quadric_decimation,
            subdivide_loop,
            labels_to_centroids,
            sample_points_uniformly,
            sample_points_poisson_disk,
            voxel_down_sample,
            points_to_labels,
            points_to_convex_hull_surface,
            surface_from_point_cloud_alpha_shape,
            surface_from_point_cloud_ball_pivoting,
            label_to_surface,
            largest_label_to_surface,
            add_quality,
            add_curvature_scalars,
            add_spherefitted_curvature]


def add_curvature(*args, **kwargs):
    warnings.warn("nppas.add_curvature is deprecated. Use add_curvature_scalars instead!")
    return add_curvature_scalars(*args, **kwargs)

def spherefitted_curvature(*args, **kwargs):
    warnings.warn("nppas.spherefitted_curvature is deprecated. Use `add_quality(..., Quality.SPHERE_FITTED_CURVATURE_..._VOXEL)` instead!")
    return add_spherefitted_curvature(*args, **kwargs)


def _vedo_stanford_bunny_layerdatatuple():
    return [(_vedo_stanford_bunny(), {}, "surface")]

def _vedo_ellipsoid_layerdatatuple():
    return [(_vedo_ellipsoid(), {}, "surface")]

def _vedo_gastruloid_layerdatatuple():
    return [(gastruloid(), {}, "surface")]


@napari_hook_implementation
def napari_provide_sample_data():
    return {
        "Standford bunny (nppas)": _vedo_stanford_bunny_layerdatatuple,
        "Ellipsoid (nppas)": _vedo_ellipsoid_layerdatatuple,
        "Gastruloid (AV Luque and JV Veenvliet (2023), nppas)": _vedo_gastruloid_layerdatatuple
    }


def gastruloid() -> "napari.types.SurfaceData":
    print ("The nppas gastruloid example is derived from AV Luque and JV Veenvliet (2023) which is licensed CC-BY (https://creativecommons.org/licenses/by/4.0/legalcode) and can be downloaded from here: https://zenodo.org/record/7603081")
    import vedo
    from pathlib import Path
    data = str(Path(__file__).parent / "data" / "gastruloid.ply")
    return to_napari_surface_data(vedo.Mesh(data))


def _vedo_ellipsoid() -> "napari.types.SurfaceData":
    import vedo
    shape = vedo.shapes.Ellipsoid().scale(10)
    return (shape.points(), np.asarray(shape.faces()))


def _vedo_stanford_bunny() -> "napari.types.SurfaceData":
    import vedo
    from pathlib import Path
    data = str(Path(__file__).parent / "data" / "bun_zipper.ply")
    return isotropic_scale_surface(to_napari_surface_data(vedo.Mesh(data)), 100)

@register_function(menu="Points > Create points from labels centroids (nppas)")
def labels_to_centroids(labels_data:"napari.types.LabelsData", viewer:"napari.Viewer" = None) -> "napari.types.PointsData":
    """Determine centroids from all labels and store them as points.

    Parameters
    ----------
    labels_data:napari.types.LabelsData
    """
    from skimage.measure import regionprops

    statistics = regionprops(labels_data)
    centroids = [s.centroid for s in statistics]
    return centroids


@register_function(menu="Points > Points to labels (nppas)")
@register_function(menu="Segmentation / labeling > Create labels from points (nppas)")
@stackview.jupyter_displayable_output
@time_slicer
def points_to_labels(points_data:"napari.types.PointsData", as_large_as_image:"napari.types.ImageData", viewer:"napari.Viewer"=None) -> "napari.types.LabelsData":
    """Mark single pixels in a zero-value pixel image if there is a point in a given point list.
    Point with index 0 in the list will get pixel intensity 1.
    If there are multiple points where the rounded coordinate is within the same pixel,
    some will be overwritten. There is no constraint which will be overwritten.

    Parameters
    ----------
    points_data:napari.types.PointsData
    as_large_as_image:napari.types.ImageData
        An image to specify the size of the output image. This image will not be overwritten.
    """

    labels_stack = np.zeros(as_large_as_image.shape, dtype=int)
    for i, p in enumerate(points_data):
        if len(labels_stack.shape) == 3:
            labels_stack[int(p[0] + 0.5), int(p[1] + 0.5), int(p[2] + 0.5)] = i + 1
        elif len(labels_stack.shape) == 2:
            labels_stack[int(p[0] + 0.5), int(p[1] + 0.5)] = i + 1
        else:
            raise NotImplementedError("Points to labels only supports 2D and 3D data")
            break

    return labels_stack


@register_function(menu="Surfaces > Surface to binary volumne (nppas)")
@register_function(menu="Segmentation / binarization > Create binary volume from surface (nppas)")
@stackview.jupyter_displayable_output
@time_slicer
def surface_to_binary_volume(surface: "napari.types.SurfaceData", as_large_as_image: "napari.types.ImageData" = None,
                     viewer: "napari.Viewer" = None) -> "napari.types.LabelsData":
    """Render a closed surface as binary image with the same size as a specified image.

    Notes
    -----
    * The outlines of the binary volume are subject to numeric rounding issues and may not be voxel-perfect.

    See Also
    --------
    * [1] https://vedo.embl.es/autodocs/content/vedo/mesh.html#vedo.mesh.Mesh.binarize
    * [2] https://vedo.embl.es/autodocs/content/vedo/volume.html#vedo.volume.BaseVolume.tonumpy

    Parameters
    ----------
    surface: napari.types.SurfaceData
    as_large_as_image: ImageData
    viewer: napari.Viewer, optional

    Returns
    -------
    binary_image:ImageData
    """
    import vedo

    my_mesh = vedo.mesh.Mesh((surface[0], surface[1]))
    vertices = my_mesh.points()  # get coordinates of surface vertices

    # get bounding box of mesh
    boundaries_l = np.min(vertices + 0.5, axis=0).astype(int)
    boundaries_r = np.max(vertices + 0.5, axis=0).astype(int)

    # replace region within bounding box with binary image
    if as_large_as_image is not None:
        binary_image = np.zeros_like(as_large_as_image, dtype=int)
        binary_image[boundaries_l[0] : boundaries_r[0],
                    boundaries_l[1] : boundaries_r[1],
                    boundaries_l[2] : boundaries_r[2]] = my_mesh.binarize().tonumpy()
    else:
        binary_image = my_mesh.binarize().tonumpy().astype(int)

    return np.asarray(binary_image > 0).astype(int)


@register_function(menu="Surfaces > Create surface from any label (marching cubes, scikit-image, nppas)")
@time_slicer
def label_to_surface(labels: "napari.types.LabelsData", label_id: int = 1) -> "napari.types.SurfaceData":
    """
    Turn a single label out of a label image into a surface using the marching cubes algorithm

    Parameters
    ----------
    labels_data:napari.types.LabelsData
    label_id: int
    """
    from skimage.measure import marching_cubes

    binary = np.asarray(labels == label_id)

    vertices, faces, normals, values = marching_cubes(binary, 0)

    return remove_duplicate_vertices(SurfaceTuple((vertices, faces, values)))


@register_function(menu="Surfaces > Create surface from all labels (marching cubes, scikit-image, nppas)")
@time_slicer
def all_labels_to_surface(labels: "napari.types.LabelsData") -> "napari.types.SurfaceData":
    """
    Turn a set of labels into a surface using the marching cubes algorithm

    Parameters
    ----------
    labels_data:napari.types.LabelsData
    """
    import vedo
    from skimage.measure import marching_cubes

    # convert to numpy in case it's not (clesperanto, dask, ...)
    labels = np.asarray(labels)

    # Create a surface for every label
    mesh_list = []
    for label in np.unique(labels)[:-1]:
        verts, faces, normals, values = marching_cubes(labels==label)
        mesh = vedo.mesh.Mesh((verts, faces))
        mesh_list.append(mesh)
    
    # merge the meshes; label is stored in `mesh.pointdata['OriginalMeshID']`
    mesh = vedo.merge(mesh_list, flag=True)

    return to_napari_surface_data(mesh)
    #(mesh.points(), np.asarray(mesh.faces()), mesh.pointdata['OriginalMeshID'])

# alias
marching_cubes = all_labels_to_surface

@register_function(menu="Surfaces > Create surface from largest label (marching cubes, scikit-image, nppas)")
@time_slicer
def largest_label_to_surface(labels: "napari.types.LabelsData") -> "napari.types.SurfaceData":
    """
    Turn the largest label in a label image into a surface using the marching cubes algorithm

    Parameters
    ----------
    labels_data:napari.types.LabelsData
    """
    from skimage.measure import regionprops
    statistics = regionprops(labels)

    label_index = np.argmax([r.area for r in statistics])
    labels_list = [r.label for r in statistics]
    label = labels_list[label_index]

    return label_to_surface(labels, label)


##################################################################################
# Deprecated functions

def _knot_mesh() -> "napari.types.SurfaceData":
    warnings.warn("nppas._knot_mesh() is deprecated. ")
    if not _check_open3d():
        return

    import open3d
    from pathlib import Path
    data = str(Path(__file__).parent / "data" / "knot.ply")
    return isotropic_scale_surface(to_surface(open3d.io.read_triangle_mesh(data)), 0.1)

def _standford_bunny() -> "napari.types.SurfaceData":
    warnings.warn("nppas._standford_bunny() is deprecated. Use nppas._vedo_stanford_bunny() instead")
    if not _check_open3d():
        return

    import open3d
    from pathlib import Path
    data = str(Path(__file__).parent / "data" / "bun_zipper.ply")
    return isotropic_scale_surface(to_surface(open3d.io.read_triangle_mesh(data)), 100)



# @register_action(menu = "Surfaces > Example data: Knot (open3d, nppas)")
def example_data_knot(viewer:"napari.Viewer"):
    warnings.warn("nppas.example_data_knot() is deprecated. ")
    viewer.add_surface(_knot_mesh(), blending='additive', shading='smooth')


# @register_action(menu = "Surfaces > Example data: Standford bunny (nppas)")
def example_data_standford_bunny(viewer:"napari.Viewer"):
    warnings.warn("nppas.example_data_standford_bunny() is deprecated. Use nppas._vedo_stanford_bunny() instead")
    viewer.add_surface(_standford_bunny(), blending='additive', shading='smooth')

# @register_action(menu = "Surfaces > Example data: Ellipsoid (vedo, nppas)")
def example_data_vedo_ellipsoid(viewer:"napari.Viewer"):
    warnings.warn("nppas.example_data_vedo_ellipsoid() is deprecated. Use nppas.vedo_example_ellipsoid() instead")
    viewer.add_surface(_vedo_ellipsoid(), blending='additive', shading='smooth')


def to_vector_d(data):
    warnings.warn("nppas.to_vector_d() is deprecated.", DeprecationWarning)
    if not _check_open3d():
        return

    import open3d
    return open3d.utility.Vector3dVector(data)


def to_vector_i(data):
    warnings.warn("nppas.to_vector_i() is deprecated.", DeprecationWarning)
    if not _check_open3d():
        return

    import open3d
    return open3d.utility.Vector3iVector(data)


def to_vector_double(data):
    warnings.warn("nppas.to_vector_double() is deprecated.", DeprecationWarning)
    if not _check_open3d():
        return

    import open3d
    return open3d.utility.DoubleVector(data)


def to_numpy(data):
    warnings.warn("nppas.to_numpy() is deprecated. Use np.asarray() instead.", DeprecationWarning)
    return np.asarray(data)


def to_mesh(data):
    warnings.warn("nppas.to_mesh() is deprecated. Use nppas.to_vedo_mesh() instead.", DeprecationWarning)
    if not _check_open3d():
        return

    import open3d
    return open3d.geometry.TriangleMesh(to_vector_d(data[0]), to_vector_i(data[1]))


def to_point_cloud(data):
    """
    http://www.open3d.org/docs/0.9.0/tutorial/Basic/working_with_numpy.html#from-numpy-to-open3d-pointcloud
    """
    warnings.warn("nppas.to_point_cloud() is deprecated. Use nppas.to_napari_points_data() instead.", DeprecationWarning)
    if not _check_open3d():
        return

    import open3d
    pcd = open3d.geometry.PointCloud()
    pcd.points = to_vector_d(data)
    return pcd


def to_surface(mesh):
    warnings.warn("nppas.to_surface() is deprecated. Use nppas.to_napari_surface_data() instead.", DeprecationWarning)
    vertices = to_numpy(mesh.vertices)
    faces = to_numpy(mesh.triangles)
    values = np.ones((vertices.shape[0]))

    return (vertices, faces, values)


# @register_function(menu="Surfaces > Convex hull (open3d, nppas)")
def convex_hull(surface:"napari.types.SurfaceData") -> "napari.types.SurfaceData":
    """Produce the convex hull surface around a surface
    """
    warnings.warn("nppas.convex_hull() is deprecated. Use nppas.convex_hull_from_surface() instead.", DeprecationWarning)
    mesh = to_mesh(surface)

    new_mesh, _ = mesh.compute_convex_hull()
    return to_surface(new_mesh)


# @register_function(menu="Surfaces > Smoothing (simple, open3d, nppas)")
def filter_smooth_simple(surface:"napari.types.SurfaceData", number_of_iterations: int = 1) -> "napari.types.SurfaceData":
    """Smooth a surface using an average filter

    Parameters
    ----------
    surface:napari.types.SurfaceData
    number_of_iterations:int

    See Also
    --------
    ..[0] http://www.open3d.org/docs/0.12.0/tutorial/geometry/mesh.html#Average-filter
    """
    warnings.warn("nppas.filter_smooth_simple() is deprecated. Use nppas.smooth_surface() instead.", DeprecationWarning)

    mesh_in = to_mesh(surface)
    mesh_out = mesh_in.filter_smooth_simple(number_of_iterations=number_of_iterations)
    return to_surface(mesh_out)


# @register_function(menu="Surfaces > Smoothing (Laplacian, open3d, nppas)")
def filter_smooth_laplacian(surface:"napari.types.SurfaceData", number_of_iterations: int = 1) -> "napari.types.SurfaceData":
    """Smooth a surface using the Laplacian method

    Parameters
    ----------
    surface:napari.types.SurfaceData
    number_of_iterations:int

    See Also
    --------
    ..[0] http://www.open3d.org/docs/0.12.0/tutorial/geometry/mesh.html#Laplacian
    """
    warnings.warn("nppas.filter_smooth_laplacian() is deprecated. Use nppas.smooth_surface() instead.", DeprecationWarning)

    mesh_in = to_mesh(surface)
    mesh_out = mesh_in.filter_smooth_laplacian(number_of_iterations=number_of_iterations)
    return to_surface(mesh_out)


# @register_function(menu="Surfaces > Smoothing (Taubin et al 1995., open3d, nppas)")
def filter_smooth_taubin(surface:"napari.types.SurfaceData", number_of_iterations: int = 1) -> "napari.types.SurfaceData":
    """Smooth a surface using Taubin's method

    Parameters
    ----------
    surface:napari.types.SurfaceData
    number_of_iterations:int

    See Also
    --------
    ..[0] http://www.open3d.org/docs/0.12.0/tutorial/geometry/mesh.html#Taubin-filter
    ..[1] G. Taubin: Curve and surface smoothing without shrinkage, ICCV, 1995.
    """
    warnings.warn("nppas.filter_smooth_taubin() is deprecated. Use nppas.smooth_surface() instead.", DeprecationWarning)

    mesh_in = to_mesh(surface)
    mesh_out = mesh_in.filter_smooth_taubin(number_of_iterations=number_of_iterations)
    return to_surface(mesh_out)


# @register_function(menu="Surfaces > Simplify using vertex clustering (open3d, nppas)")
def simplify_vertex_clustering(surface:"napari.types.SurfaceData", voxel_size: float = 5) -> "napari.types.SurfaceData":
    """Simplify a surface using vertex clustering

    Parameters
    ----------
    surface:napari.types.SurfaceData
    voxel_size:float

    See Also
    --------
    ..[0] http://www.open3d.org/docs/0.12.0/tutorial/geometry/mesh.html#Vertex-clustering
    """
    warnings.warn("nppas.simplify_vertex_clustering() is deprecated. Use nppas.decimate_quadric() or nppas.decimate_pro() instead.", DeprecationWarning)
    if not _check_open3d():
        return

    import open3d
    mesh_in = to_mesh(surface)

    mesh_out = mesh_in.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=open3d.geometry.SimplificationContraction.Average
    )
    return to_surface(mesh_out)


# @register_function(menu="Surfaces > Simplify using quadratic decimation (open3d, nppas)")
def simplify_quadric_decimation(surface:"napari.types.SurfaceData", target_number_of_triangles: int = 500) -> "napari.types.SurfaceData":
    """Simplify a surface using quadratic decimation

    Parameters
    ----------
    surface:napari.types.SurfaceData
    target_number_of_triangles:int

    See Also
    --------
    ..[0] http://www.open3d.org/docs/0.12.0/tutorial/geometry/mesh.html#Mesh-decimation
    """
    warnings.warn("nppas.simplify_quadric_decimation() is deprecated. Use nppas.decimate_quadric() instead.", DeprecationWarning)

    mesh_in = to_mesh(surface)
    mesh_out = mesh_in.simplify_quadric_decimation(target_number_of_triangles=target_number_of_triangles)
    return to_surface(mesh_out)


# @register_function(menu="Surfaces > Subdivide loop (open3d, nppas)")
def subdivide_loop(surface:"napari.types.SurfaceData", number_of_iterations: int = 1) -> "napari.types.SurfaceData":
    """Make a mesh more detailed by subdividing in a loop.
    If iterations are high, this can take very long.

    Parameters
    ----------
    surface:napari.types.SurfaceData
    number_of_iterations:int

    See Also
    --------
    ..[0] http://www.open3d.org/docs/0.12.0/tutorial/geometry/mesh.html#Mesh-subdivision
    """
    warnings.warn("nppas.subdivide_loop() is deprecated. Use nppas.subdivide_loop_vedo() instead.", DeprecationWarning)

    mesh_in = to_mesh(surface)
    mesh_out = mesh_in.subdivide_loop(number_of_iterations=number_of_iterations)
    return to_surface(mesh_out)


# @register_function(menu="Points > Create points from surface sampling uniformly (open3d, nppas)")
def sample_points_uniformly(surface:"napari.types.SurfaceData", number_of_points: int = 500, viewer:"napari.Viewer"=None) -> "napari.types.PointsData":
    """Sample points uniformly

    Parameters
    ----------
    surface:napari.types.SurfaceData
    number_of_points:int

    See Also
    --------
    ..[0] http://www.open3d.org/docs/0.12.0/tutorial/geometry/mesh.html#Sampling
    """
    warnings.warn("nppas.sample_points_uniformly() is deprecated. Use nppas.sample_points_from_surface() instead.", DeprecationWarning)

    mesh_in = to_mesh(surface)
    point_cloud = mesh_in.sample_points_uniformly(number_of_points=number_of_points)

    result = to_numpy(point_cloud.points)
    return result


# @register_function(menu="Points > Create points from surface using Poisson disk sampling (open3d, nppas)")
def sample_points_poisson_disk(surface:"napari.types.SurfaceData", number_of_points: int = 500, init_factor: float = 5, viewer:"napari.Viewer"=None) -> "napari.types.PointsData":
    """Sample a list of points from a surface using the Poisson disk algorithm

    Parameters
    ----------
    surface:napari.types.SurfaceData
    number_of_points:int
    init_factor:float

    See Also
    --------
    ..[0] http://www.open3d.org/docs/0.12.0/tutorial/geometry/mesh.html#Sampling
    """
    warnings.warn("nppas.sample_points_poisson_disk() is deprecated. Use nppas.sample_points_from_surface() instead.",
                  DeprecationWarning)

    mesh_in = to_mesh(surface)
    point_cloud = mesh_in.sample_points_poisson_disk(number_of_points=number_of_points, init_factor=init_factor)

    result = to_numpy(point_cloud.points)
    return result


# @register_function(menu="Points > Down-sample (open3d, nppas)")
def voxel_down_sample(points_data:"napari.types.PointsData", voxel_size: float = 5, viewer:"napari.Viewer" = None) -> "napari.types.PointsData":
    """Removes points from a point cloud so that the remaining points lie within a grid of
    defined voxel size.

    http://www.open3d.org/docs/0.12.0/tutorial/geometry/pointcloud.html#Voxel-downsampling
    """
    warnings.warn(
        "nppas.voxel_down_sample() is deprecated. Use nppas.subsample_points() instead.",
        DeprecationWarning)

    point_cloud = to_point_cloud(points_data)
    new_point_cloud = point_cloud.voxel_down_sample(voxel_size)

    result = to_numpy(new_point_cloud.points)
    return result


# @register_function(menu="Surfaces > Convex hull of points (open3d, nppas)")
def points_to_convex_hull_surface(points_data:"napari.types.PointsData") -> "napari.types.SurfaceData":
    """Determine the convex hull surface of a list of points

    Parameters
    ----------
    points_data:napari.types.PointsData

    See Also
    --------
    ..[0] http://www.open3d.org/docs/0.12.0/tutorial/geometry/pointcloud.html#Convex-hull
    """
    warnings.warn("nppas.points_to_convex_hull_surface() is deprecated. Use nppas.create_convex_hull_from_points() instead.", DeprecationWarning)

    point_cloud = to_point_cloud(points_data)
    mesh_out, _ = point_cloud.compute_convex_hull()

    return to_surface(mesh_out)


# @register_function(menu="Surfaces > Create surface from points (alpha-shape, open3d, nppas)")
def surface_from_point_cloud_alpha_shape(points_data:"napari.types.PointsData", alpha:float = 5) -> "napari.types.SurfaceData":
    """Turn point into a surface using alpha shapes

    Parameters
    ----------
    points_data:napari.types.PointsData
    alpha:float

    See Also
    --------
    ..[0] http://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html#Alpha-shapes
    """
    warnings.warn("nppas.surface_from_point_cloud_alpha_shape() is deprecated. Use nppas.create_convex_hull_from_points() instead.", DeprecationWarning)
    if not _check_open3d():
        return

    import open3d
    pcd = to_point_cloud(points_data)
    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    return to_surface(mesh)


# @register_function(menu="Surfaces > Create surface from points (ball-pivoting, open3d, nppas)")
def surface_from_point_cloud_ball_pivoting(points_data:"napari.types.PointsData", radius: float = 5, delta_radius=0) -> "napari.types.SurfaceData":
    """Turn point into a surface using ball pivoting

    Parameters
    ----------
    points_data:napari.types.PointsData
    radius:float
        ball radius
    delta_radius:float, optional
        if specified, radii = [radius - delta_radius, radius, radius + delta_radius] will
        be used as ball radii

    See Also
    --------
    ..[0] http://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html#Ball-pivoting
    ..[1] http://www.open3d.org/docs/0.7.0/tutorial/Basic/pointcloud.html#point-cloud
    """
    warnings.warn("nppas.surface_from_point_cloud_ball_pivoting() is deprecated. Use nppas.create_convex_hull_from_points() instead.", DeprecationWarning)
    if not _check_open3d():
        return

    import open3d
    pcd = to_point_cloud(points_data)

    pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
                                                                              max_nn=30))
    if delta_radius == 0:
        radii = [radius]
    else:
        radii = [radius - delta_radius, radius, radius + delta_radius]

    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, to_vector_double(radii))
    return to_surface(mesh)


# @register_function(menu="Surfaces > Fill holes (vedo, nppas)")
def fill_holes(surface: "napari.types.SurfaceData", size_limit: float = 100) -> "napari.types.SurfaceData":
    """
    Fill holes in a surface up to a specified size.

    Parameters
    ----------
    surface : napari.layers.Surface
    size_limit : float, optional
        Size limit to hole-filling. The default is 100.

    See also
    --------
    ..[0] https://vedo.embl.es/autodocs/content/vedo/mesh.html#vedo.mesh.Mesh.fillHoles
    """
    warnings.warn(
        "nppas.fill_holes() is deprecated. Use nppas.fill_holes_in_surface() instead.",
        DeprecationWarning)

    import vedo

    mesh = vedo.mesh.Mesh((surface[0], surface[1]))
    mesh.fill_holes(size=size_limit)

    return (mesh.points(), np.asarray(mesh.faces()))

def _check_open3d():
    try:
        import open3d
        return True
    except:
        warnings.warn("Open3D is not installed. Follow the instructions here: http://www.open3d.org/docs/release/introduction.html#python-quick-start")
    return False
