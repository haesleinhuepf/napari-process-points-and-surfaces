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
def create_convex_hull_from_surface(surface: "napari.types.SurfaceData", viewer: "napari.Viewer" = None) -> "napari.types.SurfaceData":
    """Determine the convex hull of a surface

    Parameters
    ----------
    surface:napari.types.SurfaceData
    viewer : napari.Viewer, optional
        makes light follow the camera in the given viewer


    See Also
    --------
    ..[0] https://vedo.embl.es/autodocs/content/vedo/shapes.html#vedo.shapes.ConvexHull
    """
    from ._utils import _init_viewer
    _init_viewer(viewer)

    mesh = to_vedo_mesh(surface)

    import vedo
    convex_hull_mesh = vedo.shapes.ConvexHull(mesh)

    return to_napari_surface_data(convex_hull_mesh)


@register_function(menu="Surfaces > Remove duplicate vertices (vedo, nppas)")
def remove_duplicate_vertices(surface: "napari.types.SurfaceData", viewer: "napari.Viewer" = None) -> "napari.types.SurfaceData":
    """
    Clean a surface mesh (i.e., remove duplicate faces & vertices).

    See Also
    --------
    ..[0] https://vedo.embl.es/docs/vedo/pointcloud.html#Points.clean
    
    """
    from ._utils import _init_viewer
    _init_viewer(viewer)

    mesh = to_vedo_mesh(surface)
    clean_mesh = mesh.clean()

    return to_napari_surface_data(clean_mesh)


@register_function(menu="Surfaces > Connected components labeling (vedo, nppas)")
def connected_component_labeling(surface: "napari.types.SurfaceData", viewer: "napari.Viewer" = None) -> "napari.types.SurfaceData":
    """
    Determine the connected components of a surface mesh.

    See Also
    --------
    ..[0] https://vedo.embl.es/docs/vedo/mesh.html#Mesh.compute_connectivity
    """
    from ._quantification import set_vertex_values
    from ._utils import _init_viewer
    _init_viewer(viewer)

    mesh = to_vedo_mesh(surface)
    mesh.compute_connectivity()
    region_id = mesh.pointdata["RegionId"]

    mesh_out = to_napari_surface_data(mesh)
    mesh_out = set_vertex_values(mesh_out, region_id)

    return mesh_out


@register_function(menu="Surfaces > Smooth (moving least squares, vedo, nppas)")
def smooth_surface_moving_least_squares_2d(surface: "napari.types.SurfaceData",
                                           smoothing_factor: float = 0.2,
                                           viewer: "napari.Viewer" = None) -> "napari.types.SurfaceData":
    """Apply a moving least squares approach to smooth a surface

    See Also
    --------
    ..[0] https://vedo.embl.es/autodocs/content/vedo/vedo/pointcloud.html#Points.smooth_mls_2d
    """
    from ._utils import _init_viewer
    _init_viewer(viewer)

    mesh = to_vedo_mesh(surface)

    smooth_mesh = mesh.smooth_mls_2d(f=smoothing_factor)

    return to_napari_surface_data(smooth_mesh)


@register_function(menu="Surfaces > Smooth (moving least squares with radius, vedo, nppas)")
def smooth_surface_moving_least_squares_2d_radius(surface: "napari.types.SurfaceData",
                                                  smoothing_factor: float = 0.2,
                                                  radius: float = 0.2,
                                                  viewer: "napari.Viewer" = None) -> "napari.types.SurfaceData":
    """Apply a moving least squares approach to smooth a surface. 
    
    The radius is used to determine the number of points to use for the smoothing.

    See Also
    --------
    ..[0] https://vedo.embl.es/autodocs/content/vedo/vedo/pointcloud.html#Points.smooth_mls_2d
    """
    from ._utils import _init_viewer
    _init_viewer(viewer)

    mesh = to_vedo_mesh(surface)

    smooth_mesh = mesh.smooth_mls_2d(f=smoothing_factor, radius=radius)

    return to_napari_surface_data(smooth_mesh)


@register_function(menu="Points > Smooth (moving least squares, vedo, nppas)")
def smooth_pointcloud_moving_least_squares_2d(pointcloud: "napari.types.PointsData",
                                              smoothing_factor: float = 0.2,
                                              viewer: "napari.Viewer" = None) -> "napari.types.PointsData":
    """Apply a moving least squares approach to smooth a point cloud.

    See Also
    --------
    ..[0] https://vedo.embl.es/autodocs/content/vedo/vedo/pointcloud.html#Points.smooth_mls_2d
    """
    from ._utils import _init_viewer
    _init_viewer(viewer)

    points = to_vedo_points(pointcloud)

    smooth_points = points.smooth_mls_2d(f=smoothing_factor)

    return to_napari_points_data(smooth_points)


@register_function(menu="Points > Smooth (moving least squares radius, vedo, nppas)")
def smooth_pointcloud_moving_least_squares_2d_radius(pointcloud: "napari.types.PointsData",
                                                     smoothing_factor: float = 0.2,
                                                     radius: float = 2,
                                                     viewer: "napari.Viewer" = None) -> "napari.types.PointsData":
    """Apply a moving least squares approach to smooth a point cloud.

    The radius is used to determine the number of points to use for the smoothing.

    See Also
    --------
    ..[0] https://vedo.embl.es/autodocs/content/vedo/vedo/pointcloud.html#Points.smooth_mls_2d
    """
    from ._utils import _init_viewer
    _init_viewer(viewer)

    points = to_vedo_points(pointcloud)

    smooth_points = points.smooth_mls_2d(f=smoothing_factor, radius=radius)

    return to_napari_points_data(smooth_points)


@register_function(menu="Surfaces > Smooth (Windowed Sinc, vedo, nppas)")
def smooth_surface(surface: "napari.types.SurfaceData",
                   number_of_iterations: int = 15,
                   pass_band: float = 0.1,
                   edge_angle: float = 15,
                   feature_angle: float = 60,
                   boundary: bool = False,
                   viewer: "napari.Viewer" = None
                   ) -> "napari.types.SurfaceData":
    """Smooth a surface using a Windowed Sinc kernel.

    See Also
    --------
    ..[0] https://vedo.embl.es/docs/vedo/mesh.html#Mesh.smooth
    """
    from ._utils import _init_viewer
    _init_viewer(viewer)

    mesh = to_vedo_mesh(surface)

    smooth_mesh = mesh.smooth(niter=number_of_iterations,
                              pass_band=pass_band,
                              edge_angle=edge_angle,
                              feature_angle=feature_angle,
                              boundary=boundary)

    return to_napari_surface_data(smooth_mesh)


#@register_function(menu="Surfaces > Subdivide loop (vedo, nppas)")
def _subdivide_loop_vedo(surface: "napari.types.SurfaceData",
                         number_of_iterations: int = 1,
                         viewer: "napari.Viewer" = None
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
    ..[0] https://vedo.embl.es/docs/vedo/mesh.html#Mesh.subdivide
    ..[1] https://vtk.org/doc/nightly/html/classvtkLoopSubdivisionFilter.html
    """
    from ._utils import _init_viewer
    _init_viewer(viewer)

    mesh_in = to_vedo_mesh(surface)
    mesh_out = mesh_in.subdivide(number_of_iterations, method=0)
    return to_napari_surface_data(mesh_out)


#@register_function(menu="Surfaces > Subdivide linear (vedo, nppas)")
def _subdivide_linear(surface: "napari.types.SurfaceData",
                      number_of_iterations: int = 1,
                      viewer: "napari.Viewer" = None
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
    ..[0] https://vedo.embl.es/docs/vedo/mesh.html#Mesh.subdivide
    ..[1] https://vtk.org/doc/nightly/html/classvtkLinearSubdivisionFilter.html
    """
    from ._utils import _init_viewer
    _init_viewer(viewer)

    mesh_in = to_vedo_mesh(surface)
    mesh_out = mesh_in.subdivide(number_of_iterations, method=1)
    return to_napari_surface_data(mesh_out)


@register_function(menu="Surfaces > Subdivide adaptive (vedo, nppas)")
def subdivide_adaptive(surface: "napari.types.SurfaceData",
                       number_of_iterations: int = 1,
                       maximum_edge_length: float = 0,
                       viewer: "napari.Viewer" = None
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
    ..[0] https://vedo.embl.es/docs/vedo/mesh.html#Mesh.subdivide
    ..[1] https://vtk.org/doc/nightly/html/classvtkAdaptiveSubdivisionFilter.html
    """
    from ._utils import _init_viewer
    _init_viewer(viewer)

    mesh_in = to_vedo_mesh(surface)

    if maximum_edge_length == 0:
        maximum_edge_length = mesh_in.diagonal_size(
        ) / np.sqrt(mesh_in._data.GetNumberOfPoints()) / number_of_iterations

    mesh_out = mesh_in.subdivide(
        number_of_iterations, method=2, mel=maximum_edge_length)
    return to_napari_surface_data(mesh_out)


# @register_function(menu="Surfaces > Subdivide butterfly (vedo, nppas)")
def _subdivide_butterfly(surface: "napari.types.SurfaceData",
                         number_of_iterations: int = 1,
                         viewer: "napari.Viewer" = None
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
    ..[0] https://vedo.embl.es/docs/vedo/mesh.html#Mesh.subdivide
    ..[1] https://vtk.org/doc/nightly/html/classvtkButterflySubdivisionFilter.html
    ..[2] Zorin et al. "Interpolating Subdivisions for Meshes with Arbitrary Topology," Computer Graphics Proceedings, Annual Conference Series, 1996, ACM SIGGRAPH, pp.189-192
    """
    from ._utils import _init_viewer
    _init_viewer(viewer)

    mesh_in = to_vedo_mesh(surface)
    mesh_out = mesh_in.subdivide(number_of_iterations, method=3)
    return to_napari_surface_data(mesh_out)


@register_function(menu="Surfaces > Subdivide centroid (vedo, nppas)")
def subdivide_centroid(surface: "napari.types.SurfaceData",
                         number_of_iterations: int = 1,
                         viewer: "napari.Viewer" = None
                         ) -> "napari.types.SurfaceData":
    """
    Make a mesh more detailed by centroid-based subdivision.

    Parameters
    ----------
    surface:napari.types.SurfaceData
    number_of_iterations:int, optional

    See Also
    --------
    ..[0] https://vedo.embl.es/docs/vedo/mesh.html#Mesh.subdivide
    """
    from ._utils import _init_viewer
    _init_viewer(viewer)

    mesh_in = to_vedo_mesh(surface)
    mesh_out = mesh_in.subdivide(number_of_iterations, method=4)
    return to_napari_surface_data(mesh_out)


@register_function(menu="Surfaces > Decimate surface (quadric, vedo, nppas)")
def decimate_quadric(surface: "napari.types.SurfaceData",
                     fraction: float = 0.5,
                     number_of_vertices: int = None,
                     viewer: "napari.Viewer" = None
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
    from ._utils import _init_viewer
    _init_viewer(viewer)

    mesh_in = to_vedo_mesh(surface)
    mesh_out = mesh_in.decimate(method='quadric', fraction=fraction, n=number_of_vertices)
    return to_napari_surface_data(mesh_out)


@register_function(menu="Surfaces > Decimate surface (pro, vedo, nppas)")
def decimate_pro(surface: "napari.types.SurfaceData",
                     fraction: float = 0.5,
                     number_of_vertices: int = None,
                     viewer: "napari.Viewer" = None
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
    from ._utils import _init_viewer
    _init_viewer(viewer)

    mesh_in = to_vedo_mesh(surface)
    mesh_out = mesh_in.decimate(method='pro', fraction=fraction, n=number_of_vertices)
    return to_napari_surface_data(mesh_out)


@register_function(menu="Points > Create points from surface (vedo, nppas)")
def sample_points_from_surface(surface: "napari.types.SurfaceData", distance_fraction: float = 0.01, viewer: "napari.Viewer" = None) -> "napari.types.PointsData":
    """Sample points from a surface

    Parameters
    ----------
    surface:napari.types.SurfaceData
    distance_fraction:float
        the smaller the distance, the more points

    See Also
    --------
    ..[0] https://vedo.embl.es/docs/vedo/pointcloud.html#Points.subsample
    """
    from ._utils import _init_viewer
    _init_viewer(viewer)

    mesh_in = to_vedo_mesh(surface)

    point_cloud = mesh_in.subsample(fraction=distance_fraction)

    result = to_napari_points_data(point_cloud)
    return result


@register_function(menu="Points > Subsample points (vedo, nppas)")
def subsample_points(points_data: "napari.types.PointsData", distance_fraction: float = 0.01, viewer: "napari.Viewer" = None) -> "napari.types.PointsData":
    """Subsample points

    Parameters
    ----------
    points_data:napari.types.PointsData
    distance_fraction:float
        the smaller the distance, the more points

    See Also
    --------
    ..[0] https://vedo.embl.es/docs/vedo/pointcloud.html#Points.subsample
    """
    from ._utils import _init_viewer
    _init_viewer(viewer)

    mesh_in = to_vedo_points(points_data)

    point_cloud = mesh_in.subsample(fraction=distance_fraction)

    result = to_napari_points_data(point_cloud)
    return result


@register_function(menu="Surfaces > Create surface from pointcloud (flying edges, vedo, nppas)")
def reconstruct_surface_from_pointcloud(point_cloud: "napari.types.PointsData",
                                        number_of_sampling_voxels: int = 100,
                                        point_influence_radius: float = 0.1,
                                        padding: float = 0.05,
                                        fill_holes: bool = True,
                                        viewer: "napari.Viewer" = None) -> "napari.types.SurfaceData":
    """Reconstruct a surface from a point cloud.

    Parameters
    ----------
    point_cloud:napari.types.PointsData
    number_of_sampling_voxels: int, optional
        number of voxels in each direction to sample the surface
        Can be used to control reconstruction precision.
    point_influence_radius: float, optional
        radius of influence of each point.
        Smaller values generally improve performance markedly.
    padding: float, optional
        increase by this fraction the bounding box
    fill_holes: bool, optional
        fill holes in the surface

    See Also
    --------
    ..[0] https://vedo.embl.es/docs/vedo/pointcloud.html#Points.reconstruct_surface
    """
    from ._utils import _init_viewer
    _init_viewer(viewer)

    point_cloud = to_vedo_points(point_cloud)
    mesh_out = point_cloud.reconstruct_surface(
        dims=(number_of_sampling_voxels) * 3,
        radius=point_influence_radius,
        padding=padding,
        hole_filling=fill_holes)

    mesh_out = to_napari_surface_data(mesh_out)
    if len(mesh_out[1]) == 0:
        raise ValueError("No surface could be reconstructed with the given" +
                         " parameters. Try to increase the number of " +
                         "sampling voxels or the point influence radius.")

    return mesh_out


@register_function(menu="Surfaces > Convex hull of points (vedo, nppas)")
def create_convex_hull_from_points(points_data: "napari.types.PointsData",
                                   viewer: "napari.Viewer" = None) -> "napari.types.SurfaceData":
    """Determine the convex hull surface of a list of points

    Parameters
    ----------
    points_data:napari.types.PointsData

    See Also
    --------
    ..[0] https://vedo.embl.es/autodocs/content/vedo/shapes.html#vedo.shapes.ConvexHull
    """
    import vedo
    from ._utils import _init_viewer
    _init_viewer(viewer)

    point_cloud = to_vedo_points(points_data)
    mesh_out = vedo.shapes.ConvexHull(point_cloud)

    return to_napari_surface_data(mesh_out)


@register_function(menu="Surfaces > Fill holes (vedo, nppas)")
def fill_holes_in_surface(surface: "napari.types.SurfaceData", size_limit: float = 100, viewer: "napari.Viewer" = None) -> "napari.types.SurfaceData":
    """
    Fill holes in a surface up to a specified size.

    Parameters
    ----------
    surface : napari.layers.Surface
    size_limit : float, optional
        Size limit to hole-filling. The default is 100.

    See also
    --------
    ..[0] https://vedo.embl.es/docs/vedo/mesh.html#Mesh.fillHoles
    """
    from ._utils import _init_viewer
    _init_viewer(viewer)

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
