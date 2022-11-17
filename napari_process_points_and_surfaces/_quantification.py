import warnings

from napari.types import SurfaceData, PointsData
from napari.types import LabelsData, LayerDataTuple

from napari_tools_menu import register_function, register_dock_widget
import numpy as np
from typing import List
from magicgui import magic_factory
from napari_tools_menu import register_function


from enum import Enum

class Quality(Enum):
    EDGE_RATIO = 0
    ASPECT_RATIO = 1
    RADIUS_RATIO = 2
    ASPECT_FROBENIUS = 3
    MED_ASPECT_FROBENIUS = 4
    MAX_ASPECT_FROBENIUS = 5
    MIN_ANGLE = 6
    COLLAPSE_RATIO = 7
    MAX_ANGLE = 8
    CONDITION = 9
    SCALED_JACOBIAN = 10
    SHEAR = 11
    RELATIVE_SIZE_SQUARED = 12
    SHAPE = 13
    SHAPE_AND_SIZE = 14
    DISTORTION = 15
    MAX_EDGE_RATIO = 16
    SKEW = 17
    TAPER = 18
    VOLUME = 19
    STRETCH = 20
    DIAGONAL = 21
    DIMENSION = 22
    ODDY = 23
    SHEAR_AND_SIZE = 24
    JACOBIAN = 25
    WARPAGE = 26
    ASPECT_GAMMA = 27
    AREA = 28
    ASPECT_BETA = 29
    
class Curvature(Enum):
    Gauss_Curvature = 0
    Mean_Curvature = 1
    Maximum_Curvature = 2
    Minimum_Curvature = 3

@register_function(menu="Measurement > Surface quality (vedo, nppas)")
def add_quality(surface: SurfaceData, quality_id: Quality = Quality.MIN_ANGLE) -> SurfaceData:
    import vedo
    mesh = vedo.mesh.Mesh((surface[0], surface[1]))
    if isinstance(quality_id, int):
        mesh.compute_quality(quality_id)
    else:
        mesh.compute_quality(quality_id.value)

    #print(mesh.celldata.keys())
    mesh2 = mesh.map_cells_to_points()
    #print(mesh2.pointdata.keys())

    vertices = np.asarray(mesh2.points())
    faces = np.asarray(mesh2.faces())
    values = np.asarray(mesh2.pointdata[mesh2.pointdata.keys()[0]])

    return (vertices, faces, values)


# @register_function(menu="Measurement > Surface quality table (vedo, nppas)", quality=dict(widget_type='Select', choices=Quality))
@register_dock_widget(menu="Measurement > Surface quality table (vedo, nppas)")
@magic_factory(qualities=dict(widget_type='Select', choices=Quality))
def _surface_quality_table(surface: SurfaceData, qualities:Quality = [Quality.AREA, Quality.MIN_ANGLE, Quality.MAX_ANGLE, Quality.ASPECT_RATIO], napari_viewer:"napari.Viewer" = None):
    return surface_quality_table(surface, qualities, napari_viewer)

def surface_quality_table(surface: SurfaceData, qualities, napari_viewer: "napari.Viewer" = None):
    """Produces a table of specified measurements and adds it to the napari viewer (if given)

    Parameters
    ----------
    surface: napari.types.SurfaceData
    qualities: list of Quality
    napari_viewer: napari.Viewer

    Returns
    -------
    pandas.DataFrame in case napari_viewer is None
    """
    import pandas

    table = {}
    for quality in qualities:
        result = add_quality(surface, quality)
        values = result[2]
        if len(table.keys()) == 0:
            table["vertex_index"] = list(range(len(values)))
        table[str(quality)] = values

    if napari_viewer is not None:
        # Store results in the properties dictionary:
        from napari_workflows._workflow import _get_layer_from_data
        surface_layer = _get_layer_from_data(napari_viewer, surface)
        # todo: this needs to be changed once surface layes support properties/features
        # https://github.com/napari/napari/issues/5205
        surface_layer.properties = table
        surface_layer.features = pandas.DataFrame(table)

        # turn table into a widget
        from napari_skimage_regionprops import add_table
        add_table(surface_layer, napari_viewer)
    else:
        return pandas.DataFrame(table)


@register_function(menu="Measurement > Surface quality/annotation to table (nppas)")
def surface_quality_to_properties(surface: SurfaceData,
                                  napari_viewer: "napari.Viewer",
                                  column_name: str = "annotation"):
    """Reads from an existing surface data/layer if values are present and stores the
    values in the properties/features of the layer.

    Parameters
    ----------
    surface: SurfaceData
    napari_viewer: napari.Viewer
    column_name: str

    Returns
    -------

    """
    import pandas
    # Store results in the properties dictionary:
    from napari_workflows._workflow import _get_layer_from_data
    surface_layer = _get_layer_from_data(napari_viewer, surface)

    table = {}
    if hasattr(surface_layer, "features"):
        if surface_layer.features is not None:
            table = surface_layer.features

    values = surface[2]

    if len(table.keys()) == 0:
        table["vertex_index"] = list(range(len(values)))
    table[column_name] = values

    table = pandas.DataFrame(table)

    # save results back
    # todo: update this as soon as napari Surface layers support properties/features.
    # https://github.com/napari/napari/issues/5205
    surface_layer.properties = table.to_dict(orient="list")
    surface_layer.features = table

    # turn table into a widget
    from napari_skimage_regionprops import add_table
    add_table(surface_layer, napari_viewer)


@register_function(menu="Measurement > Surface curvature (vedo, nppas)")
def add_curvature_scalars(surface: SurfaceData,
                          curvature_id: Curvature = Curvature.Gauss_Curvature,
                          ) -> SurfaceData:
    """
    Determine the surface curvature using vedo built-in functions.
    
    This function determines surface curvature using the built-in methods of
    the vedo library.

    Parameters
    ----------
    surface : SurfaceData
        3-Tuple of (points, faces, values)
    curvature_id : Union[Curvature, int] optional
        Method to be used: 0-gaussian, 1-mean, 2-max, 3-min curvature. The
        default is 0 (gaussian).
    Returns
    -------
    SurfaceData
        3-tuple consisting of (points, faces, values)
        
    See also
    --------
    Vedo curvature: https://vedo.embl.es/autodocs/content/vedo/mesh.html?highlight=curvature#vedo.mesh.Mesh.addCurvatureScalars
    """
    import vedo

    mesh = vedo.mesh.Mesh((surface[0], surface[1]))    
    if isinstance(curvature_id, int):
        used_method = curvature_id
    else:
        used_method = curvature_id.value
    
    mesh.compute_curvature(method=used_method)
    values = mesh.pointdata[curvature_id.name]
    
    return (mesh.points(), np.asarray(mesh.faces()), values)

@register_function(menu="Measurement > Surface curvature (sphere-fitted, nppas)")
def add_spherefitted_curvature(surface: SurfaceData, radius: float = 1.0) -> List[LayerDataTuple]:
    """
    Determine surface curvature by fitting a sphere to every vertex.
    
    This function iterates over all verteces in a surface, retrieves all points
    in a neighborhood defined by `radius` and fits a sphere to the retrieved
    points. The ocal curvature is then defined as 1/radius**2.

    Parameters
    ----------
    surface : SurfaceData
        3-Tuple of (points, faces, values)
    radius : float, optional
        Radius within which other points of the surface will be considered
        neighbors. The default is 1.0.

    Returns
    -------
    List[LayerDataTuple]
        A list of surface data items. The items correspond to the curvature- and
        fit residue-annotated surface, respectively. 
        With each item consisting of a `(points, faces, values)` tuple, the 
        `value` variable reflects each vertice's curvature or fit residue result.'
        
    See also
    --------
    sphere-fitting curvature: https://github.com/marcomusy/vedo/blob/master/examples/advanced/measure_curvature.py
    Curvature: https://en.wikipedia.org/wiki/Gaussian_curvature
    """
    import vedo
    
    mesh = vedo.mesh.Mesh((surface[0], surface[1]))
    
    curvature = np.zeros(mesh.N())
    residues = np.zeros(mesh.N())
    for idx in range(mesh.N()):
        
        patch = vedo.pointcloud.Points(mesh.closestPoint(mesh.points()[idx], radius=radius))
        
        try:
            s = vedo.pointcloud.fit_sphere(patch)
            curvature[idx] = 1/(s.radius)**2
            residues[idx] = s.residue
        except Exception:
            curvature[idx] = 0
            residues[idx] = 0
            
    if 0 in curvature:
        raise ValueError(f"The chosen curvature radius ({radius})"
                          "was too small to calculate curvatures. Increase " 
                          "the radius to silence this error.")
        
    properties_curvature_layer = {'name': 'curvature', 'colormap': 'viridis'}
    properties_residues_layer = {'name': 'fit residues', 'colormap': 'magma'}
        
    layer1 = ((mesh.points(), np.asarray(mesh.faces()), curvature), properties_curvature_layer, 'surface')
    layer2 = ((mesh.points(), np.asarray(mesh.faces()), residues), properties_residues_layer, 'surface')
        
    return [layer1, layer2]
