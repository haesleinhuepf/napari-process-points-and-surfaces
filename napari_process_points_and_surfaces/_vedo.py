from napari_tools_menu import register_function, register_action
import numpy as np
from ._utils import isotropic_scale_surface

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
def vedo_convex_hull(surface:"napari.types.SurfaceData") -> "napari.types.SurfaceData":
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

@register_function(menu="Surfaces > Remove duplicate vertices (vedo, nppas)")
def vedo_remove_duplicate_vertices(surface: "napari.types.SurfaceData"):
    """
    Clean a surface mesh (i.e., remove duplicate faces & vertices).

    See Also
    --------
    ..[0] https://vedo.embl.es/autodocs/content/vedo/pointcloud.html#vedo.pointcloud.Points.clean
    
    """
    mesh = to_vedo_mesh(surface)
    clean_mesh = mesh.clean()

    return to_napari_surface_data(clean_mesh)


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

    smooth_mesh = mesh.smooth( niter=number_of_iterations,
                        pass_band=pass_band,
                        edge_angle=edge_angle,
                        feature_angle=feature_angle,
                        boundary=boundary)

    return to_napari_surface_data(smooth_mesh)


@register_function(menu="Surfaces > Subdivide loop (vedo, nppas)")
def vedo_subdivide_loop(surface:"napari.types.SurfaceData", number_of_iterations: int = 1) -> "napari.types.SurfaceData":
    """Make a mesh more detailed by subdividing in a loop.
    If iterations are high, this can take very long.

    Parameters
    ----------
    surface:napari.types.SurfaceData
    number_of_iterations:int

    See Also
    --------
    ..[0] https://vedo.embl.es/autodocs/content/vedo/mesh.html#vedo.mesh.Mesh.subdivide
    """
    mesh_in = to_vedo_mesh(surface)
    mesh_out = mesh_in.subdivide(number_of_iterations)
    return to_napari_surface_data(mesh_out)


@register_function(menu="Points > Create points from surface (vedo, nppas)")
def vedo_sample_points_from_surface(surface:"napari.types.SurfaceData", distance_fraction: float = 0.01) -> "napari.types.PointsData":
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
def vedo_subsample_points(points_data:"napari.types.PointsData", distance_fraction: float = 0.01) -> "napari.types.PointsData":
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
def vedo_points_to_convex_hull_surface(points_data:"napari.types.PointsData") -> "napari.types.SurfaceData":
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










