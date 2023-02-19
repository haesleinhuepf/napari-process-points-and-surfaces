from napari_tools_menu import register_function
import numpy as np

def to_vedo_mesh(surface):
    _hide_vtk_warnings()
    import vedo
    return vedo.mesh.Mesh((surface[0], surface[1]))


def to_vedo_points(points_data):
    _hide_vtk_warnings()
    import vedo
    return vedo.pointcloud.Points(points_data)


# Adapted from https://jfine-python-classes.readthedocs.io/en/latest/subclass-tuple.html
class SurfaceTuple(tuple):
    zoom: float = 1
    azimuth: float = 90
    elevation: float = 0
    roll: float = 180
    cmap: str = 'viridis'
    """
    The nppas.SurfaceTuple class subclasses tuple and is thus compatible with napari.types.SurfaceData.
    It extends tuple with surface visualizations in Jupyter Notebooks.

    For more options when viewing Surfaces see nppas.show() and vedo.Plotter()
    """
    def __new__(self, x):
        return tuple.__new__(SurfaceTuple, x)

    def _repr_html_(self):
        """HTML representation of the surface object for IPython.
        Returns
        -------
        HTML text with the image and some properties.
        """
        import numpy as np
        from stackview._static_view import _png_to_html, _plt_to_png
        import matplotlib.pyplot as plt

        import vedo

        self.library_name = "nppas.SurfaceTuple"
        self.help_url = "https://github.com/haesleinhuepf/napari-process-points-and-surfaces"

        mesh: vedo.Mesh = to_vedo_mesh(self)

        # Draw mesh
        plotter = vedo.Plotter(offscreen=True, shape="1|1000")
        if len(self) > 2:
            mesh.cmap(self.cmap, self[2])
        plotter.show(mesh,
                     zoom=self.zoom,
                     azimuth=self.azimuth,
                     elevation=self.elevation,
                     roll=self.roll)
        np_rgb_image = plotter.screenshot(asarray=True)
        plt.imshow(np_rgb_image)
        # turn off axes
        frame1 = plt.gca()
        frame1.axes.xaxis.set_ticklabels([])
        frame1.axes.yaxis.set_ticklabels([])
        plt.tick_params(left=False, bottom=False)
        image = _png_to_html(_plt_to_png())

        # mesh statistics
        bounds = "<br/>".join(["{min_x:.3f}...{max_x:.3f}".format(min_x=min_x, max_x=max_x) for min_x, max_x in zip(mesh.bounds()[::2], mesh.bounds()[1::2])])
        average_size = "{size:.3f}".format(size=mesh.average_size())
        center_of_mass = ",".join(["{size:.3f}".format(size=x) for x in mesh.centerOfMass()])
        scale = ",".join(["{size:.3f}".format(size=x) for x in mesh.scale()])
        histogram = ""
        min_max = ""

        if len(self) > 2:
            # make histogram of values
            num_bins = 32
            h, _ = np.histogram(self[2], bins=num_bins)

            plt.figure(figsize=(1.8, 1.2))
            plt.bar(range(0, len(h)), h)

            # hide axis text
            # https://stackoverflow.com/questions/2176424/hiding-axis-text-in-matplotlib-plots
            # https://pythonguides.com/matplotlib-remove-tick-labels
            frame1 = plt.gca()
            frame1.axes.xaxis.set_ticklabels([])
            frame1.axes.yaxis.set_ticklabels([])
            plt.tick_params(left=False, bottom=False)

            histogram = _png_to_html(_plt_to_png())

            min_max = "<tr><td>min</td><td>" + str(np.min(self[2])) + "</td></tr>" + \
                      "<tr><td>max</td><td>" + str(np.max(self[2])) + "</td></tr>"

        help_text = "<b><a href=\"" + self.help_url + "\" target=\"_blank\">" + self.library_name + "</a></b><br/>"

        all = [
            "<table>",
            "<tr>",
            "<td>",
            image,
            "</td>",
            "<td style=\"text-align: center; vertical-align: top;\">",
            help_text,
            "<table>",
            "<tr><td>origin (z/y/x)</td><td>" + str(mesh.origin()).replace(" ", "&nbsp;") + "</td></tr>",
            "<tr><td>center of mass(z/y/x)</td><td>" + center_of_mass.replace(" ", "&nbsp;") + "</td></tr>",
            "<tr><td>scale(z/y/x)</td><td>" + scale.replace(" ", "&nbsp;") + "</td></tr>",
            "<tr><td>bounds (z/y/x)</td><td>" + str(bounds).replace(" ", "&nbsp;") + "</td></tr>",
            "<tr><td>average size</td><td>" + str(average_size) + "</td></tr>",
            "<tr><td>number of vertices</td><td>" + str(mesh.npoints) + "</td></tr>",
            "<tr><td>number of faces</td><td>" + str(len(mesh.faces())) + "</td></tr>",
            min_max,
            "</table>",
            histogram,
            "</td>",
            "</tr>",
            "</table>",
        ]

        return "\n".join(all)


def to_napari_surface_data(vedo_mesh, values=None):
    if values is None:
        return SurfaceTuple((vedo_mesh.points(), np.asarray(vedo_mesh.faces())))
    else:
        return SurfaceTuple((vedo_mesh.points(), np.asarray(vedo_mesh.faces()), values))


def to_napari_points_data(vedo_points):
    return vedo_points.points()


def _hide_vtk_warnings():
    from vtkmodules.vtkCommonCore import vtkObject
    vtkObject.GlobalWarningDisplayOff()


@register_function(menu="Surfaces > Convex hull (vedo, nppas)")
def create_convex_hull_from_surface(surface: "napari.types.SurfaceData") -> "napari.types.SurfaceData":
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
def remove_duplicate_vertices(surface: "napari.types.SurfaceData") -> "napari.types.SurfaceData":
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
def smooth_surface(surface: "napari.types.SurfaceData",
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


#@register_function(menu="Surfaces > Subdivide loop (vedo, nppas)")
def _subdivide_loop_vedo(surface: "napari.types.SurfaceData",
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


#@register_function(menu="Surfaces > Subdivide linear (vedo, nppas)")
def _subdivide_linear(surface: "napari.types.SurfaceData",
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


#@register_function(menu="Surfaces > Subdivide adaptive (vedo, nppas)")
def subdivide_adaptive(surface: "napari.types.SurfaceData",
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
def _subdivide_butterfly(surface: "napari.types.SurfaceData",
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


@register_function(menu="Surfaces > Subdivide centroid (vedo, nppas)")
def subdivide_centroid(surface: "napari.types.SurfaceData",
                         number_of_iterations: int = 1
                         ) -> "napari.types.SurfaceData":
    """
    Make a mesh more detailed by centroid-based subdivision.

    Parameters
    ----------
    surface:napari.types.SurfaceData
    number_of_iterations:int, optional

    See Also
    --------
    ..[0] https://vedo.embl.es/autodocs/content/vedo/mesh.html#vedo.mesh.Mesh.subdivide
    """
    mesh_in = to_vedo_mesh(surface)
    mesh_out = mesh_in.subdivide(number_of_iterations, method=4)
    return to_napari_surface_data(mesh_out)


@register_function(menu="Surfaces > Decimate surface (quadric, vedo, nppas)")
def decimate_quadric(surface: "napari.types.SurfaceData",
                     fraction: float = 0.5,
                     number_of_vertices: int = None
                    ) -> "napari.types.SurfaceData":
    """
    Reduce numbers of vertices of a surface to a given fraction.

    Parameters
    ----------
    surface:napari.types.SurfaceData
    fraction: float, optional
        reduce the number of vertices in the surface to the given fraction (0...1, default 0.5)
    number_of_vertices:int, optional
        overwrites fraction in case specified

    Returns
    -------
    SurfaceData

    See Also
    --------
    ..[0] https://vedo.embl.es/autodocs/content/vedo/vedo/mesh.html#Mesh.decimate
    """
    mesh_in = to_vedo_mesh(surface)
    mesh_out = mesh_in.decimate(method='quadric', fraction=fraction, n=number_of_vertices)
    return to_napari_surface_data(mesh_out)


@register_function(menu="Surfaces > Decimate surface (pro, vedo, nppas)")
def decimate_pro(surface: "napari.types.SurfaceData",
                     fraction: float = 0.5,
                     number_of_vertices: int = None
                    ) -> "napari.types.SurfaceData":
    """
    Reduce numbers of vertices of a surface to a given fraction.

    Parameters
    ----------
    surface:napari.types.SurfaceData
    fraction: float, optional
        reduce the number of vertices in the surface to the given fraction (0...1, default 0.5)
    number_of_vertices:int, optional
        overwrites fraction in case specified

    Returns
    -------
    SurfaceData

    See Also
    --------
    ..[0] https://vedo.embl.es/autodocs/content/vedo/vedo/mesh.html#Mesh.decimate
    """
    mesh_in = to_vedo_mesh(surface)
    mesh_out = mesh_in.decimate(method='pro', fraction=fraction, n=number_of_vertices)
    return to_napari_surface_data(mesh_out)


@register_function(menu="Points > Create points from surface (vedo, nppas)")
def sample_points_from_surface(surface: "napari.types.SurfaceData", distance_fraction: float = 0.01) -> "napari.types.PointsData":
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
def subsample_points(points_data: "napari.types.PointsData", distance_fraction: float = 0.01) -> "napari.types.PointsData":
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
def create_convex_hull_from_points(points_data: "napari.types.PointsData") -> "napari.types.SurfaceData":
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
def fill_holes_in_surface(surface: "napari.types.SurfaceData", size_limit: float = 100) -> "napari.types.SurfaceData":
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


def show(surface, zoom: float = 1, azimuth: float = 0, elevation: float = 0, cmap:str = 'viridis'):
    """
    Visualizes a surface mesh, e.g. in Jupyter Notebooks.

    Parameters
    ----------
    zoom: float, optional
        > 1: Zoom in
        < 1: Zoom out
    azimuth: float, optional
        angle in degrees for turning the view direction
    elevation: float, optional
        angle in degrees for turning the view direction
    cmap: str, optional
        colormap for visualization of values

    See also
    --------
    https://vedo.embl.es/autodocs/content/vedo/vedo/plotter.html#Plotter
    """

    from vedo import Plotter
    mesh = to_vedo_mesh((surface[0], surface[1]))
    if len(surface) > 2:
        mesh.cmap(cmap, surface[2])

    plt = Plotter()
    plt.show(mesh, zoom=zoom, azimuth=azimuth, elevation=elevation)
