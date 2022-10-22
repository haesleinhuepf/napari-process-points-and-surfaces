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