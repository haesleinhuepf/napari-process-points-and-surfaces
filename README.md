# napari-process-points-and-surfaces (nppas)

[![License](https://img.shields.io/pypi/l/napari-process-points-and-surfaces.svg?color=green)](https://github.com/haesleinhuepf/napari-process-points-and-surfaces/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-process-points-and-surfaces.svg?color=green)](https://pypi.org/project/napari-process-points-and-surfaces)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-process-points-and-surfaces.svg?color=green)](https://python.org)
[![tests](https://github.com/haesleinhuepf/napari-process-points-and-surfaces/workflows/tests/badge.svg)](https://github.com/haesleinhuepf/napari-process-points-and-surfaces/actions)
[![codecov](https://codecov.io/gh/haesleinhuepf/napari-process-points-and-surfaces/branch/master/graph/badge.svg)](https://codecov.io/gh/haesleinhuepf/napari-process-points-and-surfaces)
[![Development Status](https://img.shields.io/pypi/status/napari-process-points-and-surfaces.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-process-points-and-surfaces)](https://napari-hub.org/plugins/napari-process-points-and-surfaces)

Edit and analyse surface layers using [open3d](http://www.open3d.org/) in [napari].

## Usage

You find a couple of surface generation, smoothing and analysis functions in the menu `Tools > Surfaces` and `Tools > Points`. For detailed explanation of the underlying algorithms, please refer to the [open3d](http://www.open3d.org/docs/release/) documentation.
Some code snippets and the knot example data have been taken from the open3d project which is [MIT licensed](https://github.com/haesleinhuepf/napari-process-points-and-surfaces/blob/main/licenses_third_party/open3d_LICENSE).
The Standford Bunny example dataset has been taken from the [The Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3Dscanrep/).

For executing operations from Python scripts, see the [demo notebook](https://github.com/haesleinhuepf/napari-process-points-and-surfaces/blob/main/docs/demo.ipynb). There you also learn how this screenshot is made:

![img.png](https://github.com/haesleinhuepf/napari-process-points-and-surfaces/raw/main/docs/screenshot.png)

## Installation

You can install `napari-process-points-and-surfaces` via [pip]:

```
pip install napari-process-points-and-surfaces
```

## See also

There are other napari plugins with similar / overlapping functionality
* [pymeshlab](https://www.napari-hub.org/plugins/napari-pymeshlab)
* [napari-pyclesperanto-assistant](https://www.napari-hub.org/plugins/napari-pyclesperanto-assistant)

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-process-points-and-surfaces" is free and open source software

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
