import warnings

from napari_tools_menu import register_function, register_dock_widget
import numpy as np
from typing import List
from magicgui import magic_factory


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

    GAUSS_CURVATURE = 1001
    MEAN_CURVATURE = 1002
    MAXIMUM_CURVATURE = 1003
    MINIMUM_CURVATURE = 1004

    SPHERE_FITTED_CURVATURE_1_PERCENT = 2001
    SPHERE_FITTED_CURVATURE_2_PERCENT = 2002
    SPHERE_FITTED_CURVATURE_5_PERCENT = 2005
    SPHERE_FITTED_CURVATURE_10_PERCENT = 2010
    SPHERE_FITTED_CURVATURE_25_PERCENT = 2025
    SPHERE_FITTED_CURVATURE_50_PERCENT = 2050

    SPHERE_FITTED_CURVATURE_MICRO_VOXEL = 3046
    SPHERE_FITTED_CURVATURE_MILLI_VOXEL = 3047
    SPHERE_FITTED_CURVATURE_CENTI_VOXEL = 3048
    SPHERE_FITTED_CURVATURE_DECI_VOXEL = 3049
    SPHERE_FITTED_CURVATURE_VOXEL = 3050
    SPHERE_FITTED_CURVATURE_DECA_VOXEL = 3051
    SPHERE_FITTED_CURVATURE_HECTA_VOXEL = 3052
    SPHERE_FITTED_CURVATURE_KILO_VOXEL = 3053
    SPHERE_FITTED_CURVATURE_MEGA_VOXEL = 3054

# https://en.wikipedia.org/wiki/Nano-
ORDER_OF_MAGNITUDE = {
    Quality.SPHERE_FITTED_CURVATURE_MICRO_VOXEL.value : 0.000001,
    Quality.SPHERE_FITTED_CURVATURE_MILLI_VOXEL.value : 0.001,
    Quality.SPHERE_FITTED_CURVATURE_CENTI_VOXEL.value : 0.01,
    Quality.SPHERE_FITTED_CURVATURE_DECI_VOXEL.value  : 0.1,
    Quality.SPHERE_FITTED_CURVATURE_VOXEL.value       : 1,
    Quality.SPHERE_FITTED_CURVATURE_DECA_VOXEL.value  : 10,
    Quality.SPHERE_FITTED_CURVATURE_HECTA_VOXEL.value : 100,
    Quality.SPHERE_FITTED_CURVATURE_KILO_VOXEL.value  : 1000,
    Quality.SPHERE_FITTED_CURVATURE_MEGA_VOXEL.value  : 1000000
}

class Curvature(Enum):
    Gauss_Curvature = 0
    Mean_Curvature = 1
    Maximum_Curvature = 2
    Minimum_Curvature = 3


@register_function(menu="Measurement maps > Surface quality (vedo, nppas)")
def add_quality(surface: "napari.types.SurfaceData", quality_id: Quality = Quality.MIN_ANGLE) -> "napari.types.SurfaceData":
    from ._vedo import to_vedo_mesh
    import vedo
    from ._vedo import SurfaceTuple
    mesh = vedo.mesh.Mesh((surface[0], surface[1]))
    if not isinstance(quality_id, int):
        quality_id = quality_id.value

    if quality_id < 1000:
        mesh.compute_quality(quality_id)

        mesh2 = mesh.map_cells_to_points()
        values = np.asarray(mesh2.pointdata[mesh2.pointdata.keys()[0]])
    elif quality_id < 2000:
        curvature_id = quality_id - 1000
        mesh.compute_curvature(method=curvature_id)

        mesh2 = mesh.map_cells_to_points()
        values = np.asarray(mesh2.pointdata[mesh2.pointdata.keys()[0]])
    elif quality_id < 3000:
        percent = quality_id - 2000
        fraction = percent / 100
        radius = mesh.average_size() * fraction
        layer_data_tuple = _add_spherefitted_curvature(surface, radius)

        surface2 = layer_data_tuple[0][0]
        mesh2 = to_vedo_mesh(surface2)
        values = surface2[2]
    elif quality_id < 4000:
        radius = ORDER_OF_MAGNITUDE[quality_id]
        layer_data_tuple = _add_spherefitted_curvature(surface, radius)

        surface2 = layer_data_tuple[0][0]
        mesh2 = to_vedo_mesh(surface2)
        values = surface2[2]
    else:
        raise NotImplementedError(f"Quality {quality_id} is not implemented.")

    vertices = np.asarray(mesh2.points())
    faces = np.asarray(mesh2.faces())

    return SurfaceTuple((vertices, faces, values))


# @register_function(menu="Measurement > Surface quality table (vedo, nppas)", quality=dict(widget_type='Select', choices=Quality))
@register_dock_widget(menu="Measurement tables > Surface quality table (vedo, nppas)")
@magic_factory(qualities=dict(widget_type='Select', choices=Quality))
def _surface_quality_table(surface: "napari.types.SurfaceData", qualities:Quality = [Quality.AREA, Quality.MIN_ANGLE, Quality.MAX_ANGLE, Quality.ASPECT_RATIO], napari_viewer:"napari.Viewer" = None):
    return surface_quality_table(surface, qualities, napari_viewer)

def surface_quality_table(surface: "napari.types.SurfaceData", qualities, napari_viewer: "napari.Viewer" = None):
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
        # print("Measuring", quality)
        #try:
        result = add_quality(surface, quality)
        values = result[2]
        #except ValueError as e:
        #    warnings.warn(str(e))
        #    values = [np.nan] * len(surface[0])

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


@register_function(menu="Measurement tables > Surface quality/annotation to table (nppas)")
def surface_quality_to_properties(surface: "napari.types.SurfaceData",
                                  napari_viewer: "napari.Viewer",
                                  column_name: str = "annotation"):
    """Reads from an existing surface data/layer if values are present and stores the
    values in the properties/features of the layer.

    Parameters
    ----------
    surface: "napari.types.SurfaceData"
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


@register_function(menu="Measurement maps > Surface curvature (vedo, nppas)")
def add_curvature_scalars(surface: "napari.types.SurfaceData",
                          curvature_id: Curvature = Curvature.Gauss_Curvature,
                          ) -> "napari.types.SurfaceData":
    """
    Determine the surface curvature using vedo built-in functions.
    
    This function determines surface curvature using the built-in methods of
    the vedo library.

    Parameters
    ----------
    surface : "napari.types.SurfaceData"
        3-Tuple of (points, faces, values)
    curvature_id : Union[Curvature, int] optional
        Method to be used: 0-gaussian, 1-mean, 2-max, 3-min curvature. The
        default is 0 (gaussian).
    Returns
    -------
    "napari.types.SurfaceData"
        3-tuple consisting of (points, faces, values)
        
    See also
    --------
    Vedo curvature: https://vedo.embl.es/autodocs/content/vedo/mesh.html?highlight=curvature#vedo.mesh.Mesh.addCurvatureScalars
    """
    import vedo
    from ._vedo import SurfaceTuple

    mesh = vedo.mesh.Mesh((surface[0], surface[1]))    
    if isinstance(curvature_id, int):
        used_method = curvature_id
    else:
        used_method = curvature_id.value
    
    mesh.compute_curvature(method=used_method)
    values = mesh.pointdata[curvature_id.name]
    
    return SurfaceTuple((mesh.points(), np.asarray(mesh.faces()), values))

@register_function(menu="Measurement maps > Surface curvature (sphere-fitted, nppas)")
def add_spherefitted_curvature(surface: "napari.types.SurfaceData", radius: float = 1.0) -> List["napari.types.LayerDataTuple"]:
    """
    Determine surface curvature by fitting a sphere to every vertex.
    
    This function iterates over all verteces in a surface, retrieves all points
    in a neighborhood defined by `radius` and fits a sphere to the retrieved
    points. The local curvature is then defined as 1/radius**2.

    Parameters
    ----------
    surface : "napari.types.SurfaceData"
        3-Tuple of (points, faces, values)
    radius : float, optional
        Radius within which other points of the surface will be considered
        neighbors. The default is 1.0.

    Returns
    -------
    List[napari.types.LayerDataTuple]
        A list of surface data items. The items correspond to the curvature- and
        fit residue-annotated surface, respectively. 
        With each item consisting of a `(points, faces, values)` tuple, the 
        `value` variable reflects each vertice's curvature or fit residue result.'
        
    See also
    --------
    sphere-fitting curvature: https://github.com/marcomusy/vedo/blob/master/examples/advanced/measure_curvature.py
    Curvature: https://en.wikipedia.org/wiki/Gaussian_curvature
    """
    warnings.warn("add_spherefitted_curvature is deprecated. Use add_quality(..., Quality.SPHERE_FITTED_CURVATURE_..._VOXEL) instead")

    return _add_spherefitted_curvature(surface=surface, radius=radius)

def _add_spherefitted_curvature(surface: "napari.types.SurfaceData", radius: float = 1.0) -> List[
    "napari.types.LayerDataTuple"]:
    import vedo
    from ._vedo import SurfaceTuple
    
    mesh = vedo.mesh.Mesh((surface[0], surface[1]))
    
    curvature = np.zeros(mesh.npoints)
    residues = np.zeros(mesh.npoints)
    for idx in range(mesh.npoints):
        
        patch = vedo.pointcloud.Points(mesh.closest_point(mesh.points()[idx], radius=radius))
        
        try:
            s = vedo.pointcloud.fit_sphere(patch)
            curvature[idx] = 1/(s.radius)**2
            residues[idx] = s.residue
        except Exception:
            curvature[idx] = np.nan
            residues[idx] = np.nan
            
    if np.nan in curvature:
        warnings.warn(f"The chosen curvature radius ({radius})"
                          "was too small to calculate curvature in at least one point. Increase " 
                          "the radius to silence this error.")
        
    properties_curvature_layer = {'name': 'curvature', 'colormap': 'viridis'}
    properties_residues_layer = {'name': 'fit residues', 'colormap': 'magma'}
        
    layer1 = (SurfaceTuple((mesh.points(), np.asarray(mesh.faces()), curvature)), properties_curvature_layer, 'surface')
    layer2 = (SurfaceTuple((mesh.points(), np.asarray(mesh.faces()), residues)), properties_residues_layer, 'surface')
        
    return [layer1, layer2]


def set_vertex_values(surface: "napari.types.SurfaceData", values) -> "napari.types.SurfaceData":
    """
    Replace values of a surface with a given list of values

    Parameters
    ----------
    surface: napari.types.SurfaceData
        tuple of (Vertices, Faces, Values), values are optional
    values: list
        list of new values. Must have the same length as vertices

    Returns
    -------
    napari.types.SurfaceData
    """
    from ._vedo import SurfaceTuple

    num_vertices = len(surface[0])
    num_values = len(values)
    if num_vertices != num_values:
        raise ValueError(f"Number of vertices ({num_vertices}) and number of values ({num_values}) must be the same.")

    return SurfaceTuple((surface[0], surface[1], values))
