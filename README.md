# napari-process-points-and-surfaces (nppas)

[![License](https://img.shields.io/pypi/l/napari-process-points-and-surfaces.svg?color=green)](https://github.com/haesleinhuepf/napari-process-points-and-surfaces/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-process-points-and-surfaces.svg?color=green)](https://pypi.org/project/napari-process-points-and-surfaces)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-process-points-and-surfaces.svg?color=green)](https://python.org)
[![tests](https://github.com/haesleinhuepf/napari-process-points-and-surfaces/workflows/tests/badge.svg)](https://github.com/haesleinhuepf/napari-process-points-and-surfaces/actions)
[![codecov](https://codecov.io/gh/haesleinhuepf/napari-process-points-and-surfaces/branch/master/graph/badge.svg)](https://codecov.io/gh/haesleinhuepf/napari-process-points-and-surfaces)
[![Development Status](https://img.shields.io/pypi/status/napari-process-points-and-surfaces.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-process-points-and-surfaces)](https://napari-hub.org/plugins/napari-process-points-and-surfaces)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7654555.svg)](https://doi.org/10.5281/zenodo.7654555)

Process and analyze surfaces using [vedo](https://vedo.embl.es/) in [napari].

![img.png](https://github.com/haesleinhuepf/napari-process-points-and-surfaces/raw/main/docs/graphical_abstract.gif)
The nppas gastruloid example is derived from [AV Luque and JV Veenvliet (2023)](https://zenodo.org/record/7603081) which is licensed [CC-BY](https://creativecommons.org/licenses/by/4.0/legalcode) and can be downloaded from here: https://zenodo.org/record/7603081

## Usage

You find menus for surface generation, smoothing and analysis in the menu `Tools > Surfaces` and `Tools > Points`. 
For detailed explanation of the underlying algorithms, please refer to the [vedo](https://vedo.embl.es/) documentation.

For processing meshes in Python scripts, see the [demo notebook](https://github.com/haesleinhuepf/napari-process-points-and-surfaces/blob/main/docs/demo.ipynb). 
There you also learn how this screenshot is made:

![img.png](https://github.com/haesleinhuepf/napari-process-points-and-surfaces/raw/main/docs/screenshot5.png)

For performing quantitative measurements of surface in Python scripts, see the [demo notebook](https://github.com/haesleinhuepf/napari-process-points-and-surfaces/blob/main/docs/quality_measurements.ipynb). 
There you also learn how this screenshot is made:

![img.png](https://github.com/haesleinhuepf/napari-process-points-and-surfaces/raw/main/docs/screenshot6.png)

### Surface measurements and annotations

Using the menu `Tools > Measurement tables > Surface quality table (vedo, nppas)` you can derive quantiative measurements of
the vertices in a given surface layer. 

![img_1.png](https://github.com/haesleinhuepf/napari-process-points-and-surfaces/raw/main/docs/surface_measurements2.png)

To differentiate regions when analyzing those measurements it is recommended to use the menu `Tools > Surfaces > Annotate surface manually (nppas)`
after measurements have been made. This tool allows you to draw annotation label values on the surface. 
It is recommended to do activate a colorful colormap such as `hsv` before starting to draw annotations. 
Furthermore, set the maximum of the contrast limit range to the number of regions you want to annotate + 1.
Annotations can be drawn as freehand lines and circles.

![img.png](https://github.com/haesleinhuepf/napari-process-points-and-surfaces/raw/main/docs/surface_annotation2.png)

After measurements and annotations were done, you can save the annotation in the same measurement table using the menu
`Tools > Measurement tables > Surface quality/annotation to table (nppas)`

![img.png](https://github.com/haesleinhuepf/napari-process-points-and-surfaces/raw/main/docs/surface_annotation_in_table2.png)

For classifying surface vertices using machine learning, please refer to the [napari APOC](https://www.napari-hub.org/plugins/napari-accelerated-pixel-and-object-classification) documentation.

### Measurement visualization

To visualize measurements on the surface, just double-click on the table column headers.

![img.png](https://github.com/haesleinhuepf/napari-process-points-and-surfaces/raw/main/docs/quality_measurements.gif)

## Installation

You can install `napari-process-points-and-surfaces` via mamba/conda:

```
mamba install vedo=2022.4.1 napari-process-points-and-surfaces -c conda-forge
```

### Troubleshooting: Open3d installation

Since version 0.4.0, `nppas` does no longer depend on [open3d](http://www.open3d.org/). 
Some deprecated functions still use Open3d though. 
Follow the installation instructions in the [open3d documentation](http://www.open3d.org/docs/release/getting_started.htm) to install it and keep using those functions.
Also consider updating code and no longer using these deprecated functions. 
See [release notes](https://github.com/haesleinhuepf/napari-process-points-and-surfaces/releases/tag/0.4.0) for details.

## See also

There are other napari plugins with similar / overlapping functionality
* [morphometrics](https://www.napari-hub.org/plugins/morphometrics)  
* [napari-accelerated-pixel-and-object-classification](https://www.napari-hub.org/plugins/napari-accelerated-pixel-and-object-classification)
* [napari-pymeshlab](https://www.napari-hub.org/plugins/napari-pymeshlab)
* [napari-pyclesperanto-assistant](https://www.napari-hub.org/plugins/napari-pyclesperanto-assistant)
* [napari-stress](https://www.napari-hub.org/plugins/napari-stress)
* [surforama](https://github.com/cellcanvas/surforama)

And there is software for doing similar things:
* [meshlab](https://www.meshlab.net/)
* [paraview](https://www.paraview.org/)

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-process-points-and-surfaces" is free and open source software

## Acknowledgements

Some code snippets and example data were taken from the [vedo](https://vedo.embl.es/) and [open3d](http://www.open3d.org/) 
repositories and documentation. See [thirdparty licenses](https://github.com/haesleinhuepf/napari-process-points-and-surfaces/tree/main/licenses_third_party) for licensing details.
The Standford Bunny example dataset has been taken from [The Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3Dscanrep/).
The nppas gastruloid example is derived from [AV Luque and JV Veenvliet (2023)](https://zenodo.org/record/7603081) which is licensed [CC-BY](https://creativecommons.org/licenses/by/4.0/legalcode) and can be downloaded from here: https://zenodo.org/record/7603081

## Issues

If you encounter any problems, please create a thread on [image.sc] along with a detailed description and tag [@haesleinhuepf].

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/haesleinhuepf/napari-process-points-and-surfaces/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

[image.sc]: https://image.sc
[@haesleinhuepf]: https://twitter.com/haesleinhuepf
