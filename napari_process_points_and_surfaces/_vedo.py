from napari_tools_menu import register_function, register_action
import numpy as np
from ._utils import isotropic_scale_surface

def to_vedo_mesh(surface):
    _hide_vtk_warnings()
    import vedo
    return vedo.mesh.Mesh((surface[0], surface[1]))

def to_napari_surface_data(vedo_mesh, values=None):
    if values is None:
        return (vedo_mesh.points(), np.asarray(vedo_mesh.faces()))
    else:
        return (vedo_mesh.points(), np.asarray(vedo_mesh.faces()), values)

def _hide_vtk_warnings():
    from vtkmodules.vtkCommonCore import vtkObject
    vtkObject.GlobalWarningDisplayOff()


@register_function(menu="Surfaces > Convex hull (vedo, nppas)")
def vedo_convex_hull(surface:"napari.types.SurfaceData") -> "napari.types.SurfaceData":
    mesh = to_vedo_mesh(surface)

    import vedo
    convex_hull_mesh = vedo.shapes.ConvexHull(mesh)

    return to_napari_surface_data(convex_hull_mesh)

def _vedo_ellipsoid() -> "napari.types.SurfaceData":
    import vedo
    shape = vedo.shapes.Ellipsoid()
    return isotropic_scale_surface((shape.points(), np.asarray(shape.faces())), 10)

@register_action(menu = "Surfaces > Example data: Ellipsoid (vedo, nppas)")
def vedo_example_ellipsoid(viewer:"napari.viewer"):
    viewer.add_surface(_vedo_ellipsoid(), blending='additive', shading='smooth')

@register_function(menu="Surfaces > Smooth (vedo, nppas)")
def vedo_mesh_smooth(surface: "napari.types.SurfaceData",
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
    ..[0] http://www.open3d.org/docs/0.12.0/tutorial/geometry/mesh.html#Mesh-subdivision
    """
    mesh_in = to_vedo_mesh(surface)
    mesh_out = mesh_in.subdivide(number_of_iterations)
    return to_napari_surface_data(mesh_out)


