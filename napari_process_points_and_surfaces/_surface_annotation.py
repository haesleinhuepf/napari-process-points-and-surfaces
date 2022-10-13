# -*- coding: utf-8 -*-

import napari
import copy
import numpy as np

use_pygeodesic = False
try:
    from pygeodesic import geodesic
    use_pygeodesic = True
except:
    pass

from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QButtonGroup, QSpinBox
from qtpy.QtCore import QEvent, QObject

from magicgui.widgets import create_widget

from napari.layers import Surface
from scipy import spatial

from napari_tools_menu import register_dock_widget

@register_dock_widget(menu = "Surfaces > Annotate surface manually (nppas)")
class surface_annotator(QWidget):
    """Comprehensive stress analysis of droplet points layer."""

    def __init__(self, napari_viewer):
        super().__init__()

        # self.selected_vertices = []
        # # if "start_end_points" not in napari_viewer.layers :
        # #     self.points_layer = napari_viewer.add_points(np.empty((0,3)), ndim=3, name="start_end_points")
        # # else:
        # #     self.points_layer = napari_viewer.layers["start_end_points"]

        self.viewer = napari_viewer

        # add input dropdowns to plugin
        self.surface_layer_select = create_widget(annotation=Surface, label="Surface_layer")
        self.annotation_name = create_widget(annotation=str, label="Annotation name")
        self.annotation_name.value = "annotation"

        self.tool_select_group = QButtonGroup()
        self.tool_select_group.setExclusive(True)

        self.button_single_face = QPushButton("Paint Single Face")
        self.button_single_face.setCheckable(True)
        self.tool_select_group.addButton(self.button_single_face)

        self.button_radius = QPushButton("Radius")
        self.button_radius.setCheckable(True)
        self.tool_select_group.addButton(self.button_radius)

        self.button_geodesic_radius = QPushButton("geodesic Radius")
        self.button_geodesic_radius.setCheckable(True)

        self.button_erase = QPushButton('Erase annotations')

        self.tool_select_group.addButton(self.button_geodesic_radius)

        self.label_select_spinbox = QSpinBox()
        self.label_select_spinbox.setValue(2)

        self.setLayout(QVBoxLayout())

        self.layout().addWidget(self.surface_layer_select.native, 0)
        self.layout().addWidget(self.annotation_name.native)
        self.layout().addWidget(self.button_single_face)
        self.layout().addWidget(self.button_radius)
        if use_pygeodesic:
            self.layout().addWidget(self.button_geodesic_radius)
        self.layout().addWidget(self.label_select_spinbox)
        self.layout().addWidget(self.button_erase)

        self.tool_select_group.buttonClicked.connect(self.on_push_button)
        self.button_erase.clicked.connect(self._on_erase_button)
        self.installEventFilter(self)

        self.currently_selected_button = None

    def on_push_button(self, button):

        # make sure that the surface layer in the dropdown is selected in the
        # layer list when a button is clicked.
        self.viewer.layers.selection.active = self.surface_layer_select.value

        # remove previous callbacks
        if len(self.surface_layer_select.value.mouse_drag_callbacks) > 0:
            self.surface_layer_select.value.mouse_drag_callbacks.pop(0)

        # if a button is just de-activated
        if button == self.currently_selected_button:
            button.setChecked(False)
            self.surface_layer_select.value.mouse_drag_callbacks = []

        if button == self.button_single_face:
            self.surface_layer_select.value.mouse_drag_callbacks.append(self._paint_face_on_drag)
            self.currently_selected_button = button

        if button == self.button_radius:
            self.surface_layer_select.value.mouse_drag_callbacks.append(self._paint_face_by_eucledean_distance)
            self.currently_selected_button = button

        if button == self.button_geodesic_radius:
            self.surface_layer_select.value.mouse_drag_callbacks.append(self._paint_face_by_geodesic_distance)
            self.currently_selected_button = button

    def _on_erase_button(self):
        """Replace the values of a surface with zeroes."""
        data = list(self.surface_layer_select.value.data)
        data[2] = np.ones_like(data[2])
        self.surface_layer_select.value.data = data

    def eventFilter(self, obj: QObject, event: QEvent):
        """https://forum.image.sc/t/composing-workflows-in-napari/61222/3."""
        if event.type() == QEvent.ParentChange:
            self.surface_layer_select.parent_changed.emit(self.parent())

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
        layer = self.surface_layer_select.value
        visual = viewer.window._qt_window._qt_viewer.layer_to_visual[layer]

        return visual


    def paint_face(self, surface_layer, face_index: int, label: int = 0):
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
        indeces_of_triangle_points = data[1][face_index]

        values = data[2]
        values[indeces_of_triangle_points] = label

        surface_visual = self.get_napari_visual(self.viewer, surface_layer)
        meshdata = surface_visual.node._meshdata

        meshdata.set_vertex_values(values)

        surface_visual.node.set_data(meshdata=meshdata)

        if hasattr(surface_layer, "properties"):
            surface_layer.properties[self.annotation_name] = values
        if hasattr(surface_layer, "features"):
            surface_layer.features[self.annotation_name] = values

    def _paint_face_on_drag(self, layer, event):
        if "Alt" not in event.modifiers:
            return

        _, triangle_index = layer.get_value(event.position, view_direction=event.view_direction, dims_displayed=event.dims_displayed, world=True)
        self.paint_face(layer, triangle_index, self.label_select_spinbox.value())

        yield
        layer.interactive = False

        while event.type == 'mouse_move':
            _, triangle_index = layer.get_value(event.position, view_direction=event.view_direction, dims_displayed=event.dims_displayed, world=True)
            self.paint_face(layer, triangle_index, self.label_select_spinbox.value())
            yield
        layer.interactive = True



    def _paint_face_by_eucledean_distance(self, layer, event):
        if "Alt" not in event.modifiers:
            return

        click_origin = event.position
        _, triangle_index = layer.get_value(event.position, view_direction=event.view_direction, dims_displayed=event.dims_displayed, world=True)

        candidate_vertices = layer.data[1][triangle_index]
        candidate_points = layer.data[0][candidate_vertices]
        _,intersection_coords = napari.utils.geometry.find_nearest_triangle_intersection(event.position, event.view_direction, candidate_points[None, :, :])

        self.tree = spatial.KDTree(self.surface_layer_select.value.data[0])
        indices = self.tree.query_ball_point(intersection_coords, r=10)

        original_values = layer.data[2]

        yield
        layer.interactive = False
        while event.type == 'mouse_move':

            radius = np.linalg.norm(np.asarray(click_origin) - np.asarray(event.position))

            indices = self.tree.query_ball_point(intersection_coords, r=radius)
            new_values = copy.copy(original_values)
            new_values[indices] = self.label_select_spinbox.value()

            # get data, replace values and write back
            data = list(layer.data)
            data[2] = new_values
            layer.data = data

            yield
        layer.interactive = True

    def _paint_face_by_geodesic_distance(self, layer, event):
        if "Alt" not in event.modifiers:
            return

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
            new_values[indices] = self.label_select_spinbox.value()

            # get data, replace values and write back
            data = list(layer.data)
            data[2] = new_values
            self.viewer.layers[layer.name].data = data

            yield
            # the yield statement allows the mouse UI to keep working while
            # this loop is executed repeatedly
            # yield
        layer.interactive = True
