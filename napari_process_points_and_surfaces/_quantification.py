
from napari.types import SurfaceData, PointsData
from napari.types import LabelsData, LayerData

from napari_plugin_engine import napari_hook_implementation
from napari_tools_menu import register_function, register_action
import numpy as np
import napari

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

def add_quality(surface: SurfaceData, quality_id: Quality = Quality.MIN_ANGLE) -> SurfaceData:
    import vedo
    mesh = vedo.mesh.Mesh((surface[0], surface[1]))
    if isinstance(quality_id, int):
        mesh.addQuality(quality_id)
    else:
        mesh.addQuality(quality_id.value)

    #print(mesh.celldata.keys())
    mesh2 = mesh.mapCellsToPoints()
    #print(mesh2.pointdata.keys())

    vertices = np.asarray(mesh2.points())
    faces = np.asarray(mesh2.faces())
    values = np.asarray(mesh2.pointdata[mesh2.pointdata.keys()[0]])

    return (vertices, faces, values)