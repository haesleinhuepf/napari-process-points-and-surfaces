# -*- coding: utf-8 -*-

import napari
import copy
import numpy as np

_use_pygeodesic = False
try:
    from pygeodesic import geodesic
    _use_pygeodesic = True
except:
    pass

from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QButtonGroup, QSpinBox, QLabel
from qtpy.QtCore import QEvent, QObject

from magicgui.widgets import create_widget

from napari.layers import Surface
from scipy import spatial

from napari_tools_menu import register_dock_widget

@register_dock_widget(menu = "Surfaces > Annotate surface manually (nppas)")
class SurfaceAnnotationWidget(QWidget):
    """A widget for annotating surface"""

    def __init__(self, napari_viewer):
        super().__init__()

        self._viewer = napari_viewer

        # select layer
        self._surface_layer_select = create_widget(annotation=Surface, label="Surface_layer")

        # select tool
        self._tool_select_group = QButtonGroup()
        self._tool_select_group.setExclusive(True)

        self._button_off = QPushButton("Off")
        self._button_off.setCheckable(True)
        self._button_off.setChecked(True)
        self._tool_select_group.addButton(self._button_off)

        self._button_single_face = QPushButton("Freehand drawing")
        self._button_single_face.setCheckable(True)
        self._tool_select_group.addButton(self._button_single_face)

        self._button_radius = QPushButton("Draw circle")
        self._button_radius.setCheckable(True)
        self._tool_select_group.addButton(self._button_radius)

        self._button_geodesic_radius = QPushButton("Geodesic radius")
        self._button_geodesic_radius.setCheckable(True)
        self._tool_select_group.addButton(self._button_geodesic_radius)

        self._button_erase = QPushButton('Erase annotations (set all to 1)')

        # annotation configuration
        self._annotation_label_select = QSpinBox()
        self._annotation_label_select.setValue(2)

        # configure layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel("Surface layer to draw on"))
        self.layout().addWidget(self._surface_layer_select.native, 0)


        self.layout().addWidget(QLabel("Drawing mode"))
        self.layout().addWidget(self._button_off)
        self.layout().addWidget(self._button_single_face)
        self.layout().addWidget(self._button_radius)
        if _use_pygeodesic:
            self.layout().addWidget(self._button_geodesic_radius)

        self.layout().addWidget(QLabel("Annotation label"))
        self.layout().addWidget(self._annotation_label_select)
        self.layout().addWidget(self._button_erase)

        # connect events
        self._tool_select_group.buttonClicked.connect(self._on_push_button)
        self._button_erase.clicked.connect(self._on_erase_button)
        self.installEventFilter(self)

        self.currently_selected_button = None

    def _on_push_button(self, button):
        # make sure that the surface layer in the dropdown is selected in the
        # layer list when a button is clicked.
        self._viewer.layers.selection.active = self._surface_layer_select.value

        # remove previous callbacks
        while len(self._surface_layer_select.value.mouse_drag_callbacks) > 0:
            self._surface_layer_select.value.mouse_drag_callbacks.pop(0)

        # if a button is just de-activated
        if button == self.currently_selected_button or button == self._button_off:
            button.setChecked(False)
            self._surface_layer_select.value.mouse_drag_callbacks = []
            self._button_off.setChecked(True)
            button = self._button_off

        self._viewer.camera.interactive = button == self._button_off

        button_function = {
            self._button_single_face: self._paint_face_on_drag,
            self._button_radius:self._paint_face_by_euclidean_distance,
            self._button_geodesic_radius:self._paint_face_by_geodesic_distance
        }

        if button in button_function.keys():
            self._surface_layer_select.value.mouse_drag_callbacks.append(button_function[button])

        self.currently_selected_button = button

    def _on_erase_button(self):
        """Replace the values of a surface with ones. This marks all vertices as un-annotated."""
        data = list(self._surface_layer_select.value.data)
        data[2] = np.ones_like(data[2])
        self._surface_layer_select.value.data = data

    def eventFilter(self, obj: QObject, event: QEvent):
        """https://forum.image.sc/t/composing-workflows-in-napari/61222/3."""
        if event.type() == QEvent.ParentChange:
            self._surface_layer_select.parent_changed.emit(self.parent())

        return super().eventFilter(obj, event)

    def get_napari_visual(self, viewer):
        """Get the visual class for a given layer
        Parameters
        ----------
        viewer
            The napari viewer object
        layer
            The napari layer object for which to find the visual.
        Returns
        -------
        visual
            The napari visual class for the layer.
        """
        layer = self._surface_layer_select.value
        visual = viewer.window._qt_window._qt_viewer.layer_to_visual[layer]

        return visual


    def _paint_face(self, surface_layer: "napari.layer.Surface", face_index: int, annotation_label: int = 0):
        """
        Paint a face according to its index in the list of faces with a label.

        Parameters
        ----------
        surface_layer : napari.layers.Surface
        face_index : int

        Returns
        -------
        None.

        """
        data = list(surface_layer.data)
        indices_of_triangle_points = data[1][face_index]

        values = data[2]
        values[indices_of_triangle_points] = annotation_label

        surface_visual = self.get_napari_visual(self._viewer)
        meshdata = surface_visual.node._meshdata

        meshdata.set_vertex_values(values)

        surface_visual.node.set_data(meshdata=meshdata)
        self._update_contrast_limits(surface_layer, annotation_label)

    def _update_contrast_limits(self, surface_layer, annotation_label):
        """update contrast limits in case they exceed current setting"""
        if annotation_label > surface_layer.contrast_limits[1]:
            surface_layer.contrast_limits = [surface_layer.contrast_limits[0], annotation_label]

    def _paint_face_on_drag(self, layer, event):
        #if "Alt" not in event.modifiers:
        #    return

        _, triangle_index = layer.get_value(event.position, view_direction=event.view_direction, dims_displayed=event.dims_displayed, world=True)
        self._paint_face(layer, triangle_index, self._annotation_label_select.value())

        yield
        layer.interactive = False

        while event.type == 'mouse_move':
            _, triangle_index = layer.get_value(event.position, view_direction=event.view_direction, dims_displayed=event.dims_displayed, world=True)
            self._paint_face(layer, triangle_index, self._annotation_label_select.value())
            yield
        layer.interactive = True

    def _paint_face_by_euclidean_distance(self, layer, event):
        #if "Alt" not in event.modifiers:
        #    return

        click_origin = event.position
        _, triangle_index = layer.get_value(event.position, view_direction=event.view_direction, dims_displayed=event.dims_displayed, world=True)

        candidate_vertices = layer.data[1][triangle_index]
        candidate_points = layer.data[0][candidate_vertices]
        _,intersection_coords = napari.utils.geometry.find_nearest_triangle_intersection(event.position, event.view_direction, candidate_points[None, :, :])

        self.tree = spatial.KDTree(self._surface_layer_select.value.data[0])
        indices = self.tree.query_ball_point(intersection_coords, r=10)

        original_values = layer.data[2]

        yield
        layer.interactive = False
        while event.type == 'mouse_move':

            radius = np.linalg.norm(np.asarray(click_origin) - np.asarray(event.position))

            indices = self.tree.query_ball_point(intersection_coords, r=radius)
            new_values = copy.copy(original_values)
            new_values[indices] = self._annotation_label_select.value()

            # get data, replace values and write back
            data = list(layer.data)
            data[2] = new_values
            layer.data = data
            self._update_contrast_limits(layer, self._annotation_label_select.value())

            yield
        layer.interactive = True

    def _paint_face_by_geodesic_distance(self, layer, event):
        #if "Alt" not in event.modifiers:
        #    return

        click_origin = event.position
        _, triangle_index = layer.get_value(event.position, view_direction=event.view_direction, dims_displayed=event.dims_displayed, world=True)

        candidate_vertices = layer.data[1][triangle_index]
        candidate_points = layer.data[0][candidate_vertices]
        _,intersection_coords = napari.utils.geometry.find_nearest_triangle_intersection(event.position, event.view_direction, candidate_points[None, :, :])

        distances=np.linalg.norm(intersection_coords[None, :] - candidate_points, axis=1)
        index = np.argmin(distances)

        geoalg = geodesic.PyGeodesicAlgorithmExact(layer.data[0], layer.data[1])
        distances, _ = geoalg.geodesicDistances([candidate_vertices[index]], None)

        original_values = layer.data[2]

        yield
        layer.interactive = False
        while event.type == 'mouse_move':

            radius = np.linalg.norm(np.asarray(click_origin) - np.asarray(event.position))

            #indices = tree.query_ball_point(intersection_coords, r=radius)
            indices = np.argwhere(distances <= radius)

            #_, triangle_index = layer.get_value(event.position, view_direction=event.view_direction, dims_displayed=event.dims_displayed, world=True)
            #layer.data[2] = original_values
            new_values = copy.copy(original_values)
            new_values[indices] = self._annotation_label_select.value()

            # get data, replace values and write back
            data = list(layer.data)
            data[2] = new_values
            layer.data = data
            self._update_contrast_limits(layer, self._annotation_label_select.value())

            yield
            # the yield statement allows the mouse UI to keep working while
            # this loop is executed repeatedly
            # yield
        layer.interactive = True
