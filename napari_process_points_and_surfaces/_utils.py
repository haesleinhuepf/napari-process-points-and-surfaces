from napari_tools_menu import register_function

@register_function(menu = "Surfaces > Scale surface (isotropic, nppas)",
                    scale_factor={'min':0.01, 'max':100000})
def isotropic_scale_surface(surface:"napari.types.SurfaceData", scale_factor:float = 1) -> "napari.types.SurfaceData":
    """
    Scales a surface with a given factor.

    Parameters
    ----------
    surface
    scale_factor

    Returns
    -------
    surface
    """
    result = list(surface)
    result[0] = result[0] * scale_factor
    return tuple(result)