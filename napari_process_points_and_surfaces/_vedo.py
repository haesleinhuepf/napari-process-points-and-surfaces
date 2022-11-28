from napari_tools_menu import register_function, register_action
import numpy as np
from ._utils import isotropic_scale_surface
from enum import Enum


def to_vedo_mesh(surface):
    _hide_vtk_warnings()
    import vedo
    return vedo.mesh.Mesh((surface[0], surface[1]))


def to_vedo_points(points_data):
    _hide_vtk_warnings()
    import vedo
    return vedo.pointcloud.Points(points_data)


def to_napari_surface_data(vedo_mesh, values=None):
    if values is None:
        return (vedo_mesh.points(), np.asarray(vedo_mesh.faces()))
    else:
        return (vedo_mesh.points(), np.asarray(vedo_mesh.faces()), values)


def to_napari_points_data(vedo_points):
    return vedo_points.points()


def _hide_vtk_warnings():
    from vtkmodules.vtkCommonCore import vtkObject
    vtkObject.GlobalWarningDisplayOff()


@register_function(menu="Surfaces > Convex hull (vedo, nppas)")
def vedo_convex_hull(surface: "napari.types.SurfaceData") -> "napari.types.SurfaceData":
    """Determine the convex hull of a surface

    Parameters
    ----------
    surface:napari.types.SurfaceData

    See Also
    --------
    ..[0] https://vedo.embl.es/autodocs/content/vedo/shapes.html#vedo.shapes.ConvexHull
    """
    mesh = to_vedo_mesh(surface)

    import vedo
    convex_hull_mesh = vedo.shapes.ConvexHull(mesh)

    return to_napari_surface_data(convex_hull_mesh)


@register_function(menu="Surfaces > Smooth (vedo, nppas)")
def vedo_smooth_mesh(surface: "napari.types.SurfaceData",
                     number_of_iterations: int = 15,
                     pass_band: float = 0.1,
                     edge_angle: float = 15,
                     feature_angle: float = 60,
                     boundary: bool = False
                     ) -> "napari.types.SurfaceData":
    """Smooth a surface

    See Also
    --------
    ..[0] https://vedo.embl.es/autodocs/content/vedo/mesh.html#vedo.mesh.Mesh.smooth
    """

    mesh = to_vedo_mesh(surface)

    smooth_mesh = mesh.smooth(niter=number_of_iterations,
                              pass_band=pass_band,
                              edge_angle=edge_angle,
                              feature_angle=feature_angle,
                              boundary=boundary)

    return to_napari_surface_data(smooth_mesh)


class subdivision_methods(Enum):
    """Available subdivision methods"""
    loop = 0
    linear = 1
    adaptive = 2
    butterfly = 3


@register_function(menu="Surfaces > Subdivide loop (vedo, nppas)")
def vedo_subdivide_loop(surface: "napari.types.SurfaceData",
                        number_of_iterations: int = 1
                        ) -> "napari.types.SurfaceData":
    """
    Make a mesh more detailed by subdividing in a loop.

    This increases the number of faces on the surface simply by subdividing
    each triangle into four new triangles.

    Parameters
    ----------
    surface:napari.types.SurfaceData
    number_of_iterations:int

    See Also
    --------
    ..[0] https://vedo.embl.es/autodocs/content/vedo/mesh.html#vedo.mesh.Mesh.subdivide
    ..[1] https://vtk.org/doc/nightly/html/classvtkLoopSubdivisionFilter.html
    """
    mesh_in = to_vedo_mesh(surface)
    mesh_out = mesh_in.subdivide(number_of_iterations, method=0)
    return to_napari_surface_data(mesh_out)


@register_function(menu="Surfaces > Subdivide linear (vedo, nppas)")
def vedo_subdivide_linear(surface: "napari.types.SurfaceData",
                          number_of_iterations: int = 1
                          ) -> "napari.types.SurfaceData":
    """
    Make a mesh more detailed by linear subdivision.

    The position of the created triangles is determined by a
    linear interpolation method and is thus slower than the
    loop subdivision algorithm.

    Parameters
    ----------
    surface:napari.types.SurfaceData
    number_of_iterations:int

    See Also
    --------
    ..[0] https://vedo.embl.es/autodocs/content/vedo/mesh.html#vedo.mesh.Mesh.subdivide
    ..[1] https://vtk.org/doc/nightly/html/classvtkLinearSubdivisionFilter.html
    """
    mesh_in = to_vedo_mesh(surface)
    mesh_out = mesh_in.subdivide(number_of_iterations, method=1)
    return to_napari_surface_data(mesh_out)


@register_function(menu="Surfaces > Subdivide adaptive (vedo, nppas)")
def vedo_subdivide_adaptive(surface: "napari.types.SurfaceData",
                            number_of_iterations: int = 1,
                            maximum_edge_length: float = 0.
                            ) -> "napari.types.SurfaceData":
    """
    Make a mesh more detailed by adaptive subdivision.

    Each triangle is split into a set of new triangles based
    on a given maximum edge length or triangle area. If the 
    `maximum_edge_length` parameter is set to 0, then the 
    parameter will be estimated automatically.

    Parameters
    ----------
    surface:napari.types.SurfaceData
    number_of_iterations:int

    See Also
    --------
    ..[0] https://vedo.embl.es/autodocs/content/vedo/mesh.html#vedo.mesh.Mesh.subdivide
    ..[1] https://vtk.org/doc/nightly/html/classvtkAdaptiveSubdivisionFilter.html
    """
    mesh_in = to_vedo_mesh(surface)

    if maximum_edge_length == 0:
        maximum_edge_length = mesh_in.diagonal_size(
        ) / np.sqrt(mesh_in._data.GetNumberOfPoints()) / number_of_iterations

    mesh_out = mesh_in.subdivide(
        number_of_iterations, method=2, mel=maximum_edge_length)
    return to_napari_surface_data(mesh_out)


@register_function(menu="Surfaces > Subdivide butterfly (vedo, nppas)")
def vedo_subdivide_butterfly(surface: "napari.types.SurfaceData",
                             number_of_iterations: int = 1
                             ) -> "napari.types.SurfaceData":
    """
    Make a mesh more detailed by adaptive subdivision.

    Each triangle is split into a set of new triangles based
    on an 8-point butterfly scheme.

    Parameters
    ----------
    surface:napari.types.SurfaceData
    number_of_iterations:int

    See Also
    --------
    ..[0] https://vedo.embl.es/autodocs/content/vedo/mesh.html#vedo.mesh.Mesh.subdivide
    ..[1] https://vtk.org/doc/nightly/html/classvtkButterflySubdivisionFilter.html
    ..[2] Zorin et al. "Interpolating Subdivisions for Meshes with Arbitrary Topology," Computer Graphics Proceedings, Annual Conference Series, 1996, ACM SIGGRAPH, pp.189-192
    """
    mesh_in = to_vedo_mesh(surface)
    mesh_out = mesh_in.subdivide(number_of_iterations, method=3)
    return to_napari_surface_data(mesh_out)


@register_function(menu="Surfaces > Subdivide loop (vedo, nppas)")
def vedo_subdivide(surface: "napari.types.SurfaceData",
                   number_of_iterations: int = 1,
                   method: subdivision_methods = subdivision_methods.adaptive
                   ) -> "napari.types.SurfaceData":
    """Make a mesh more detailed by subdividing in a loop.
    If iterations are high, this can take very long.

    Parameters
    ----------
    surface:napari.types.SurfaceData
    number_of_iterations:int
    method: int
        Loop(0), Linear(1), Adaptive(2), Butterfly(3)
    See Also
    --------
    ..[0] https://vedo.embl.es/autodocs/content/vedo/mesh.html#vedo.mesh.Mesh.subdivide
    """
    if isinstance(method, subdivision_methods):
        method = method.value
    mesh_in = to_vedo_mesh(surface)
    mesh_out = mesh_in.subdivide(number_of_iterations, method=method)
    return to_napari_surface_data(mesh_out)


@register_function(menu="Points > Create points from surface (vedo, nppas)")
def vedo_sample_points_from_surface(surface: "napari.types.SurfaceData", distance_fraction: float = 0.01) -> "napari.types.PointsData":
    """Sample points from a surface

    Parameters
    ----------
    surface:napari.types.SurfaceData
    distance_fraction:float
        the smaller the distance, the more points

    See Also
    --------
    ..[0] https://vedo.embl.es/autodocs/content/vedo/pointcloud.html#vedo.pointcloud.Points.subsample
    """

    mesh_in = to_vedo_mesh(surface)

    point_cloud = mesh_in.subsample(fraction=distance_fraction)

    result = to_napari_points_data(point_cloud)
    return result


@register_function(menu="Points > Subsample points (vedo, nppas)")
def vedo_subsample_points(points_data: "napari.types.PointsData", distance_fraction: float = 0.01) -> "napari.types.PointsData":
    """Subsample points

    Parameters
    ----------
    points_data:napari.types.PointsData
    distance_fraction:float
        the smaller the distance, the more points

    See Also
    --------
    ..[0] https://vedo.embl.es/autodocs/content/vedo/pointcloud.html#vedo.pointcloud.Points.subsample
    """

    mesh_in = to_vedo_points(points_data)

    point_cloud = mesh_in.subsample(fraction=distance_fraction)

    result = to_napari_points_data(point_cloud)
    return result


@register_function(menu="Surfaces > Convex hull of points (vedo, nppas)")
def vedo_points_to_convex_hull_surface(points_data: "napari.types.PointsData") -> "napari.types.SurfaceData":
    """Determine the convex hull surface of a list of points

    Parameters
    ----------
    points_data:napari.types.PointsData

    See Also
    --------
    ..[0] https://vedo.embl.es/autodocs/content/vedo/shapes.html#vedo.shapes.ConvexHull
    """
    import vedo

    point_cloud = to_vedo_points(points_data)
    mesh_out = vedo.shapes.ConvexHull(point_cloud)

    return to_napari_surface_data(mesh_out)


@register_function(menu="Surfaces > Fill holes (vedo, nppas)")
def vedo_fill_holes(surface: "napari.types.SurfaceData", size_limit: float = 100) -> "napari.types.SurfaceData":
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
    mesh = to_vedo_mesh((surface[0], surface[1]))
    mesh.fill_holes(size=size_limit)

    return to_napari_surface_data(mesh)
