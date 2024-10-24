<!-- prettier ignore -->
<div align="center">
<h1> NiftyOne </h1>

![Python3](https://img.shields.io/badge/python->=3.10-blue.svg)
[![codecov](https://codecov.io/gh/childmindresearch/niftyone/branch/main/graph/badge.svg?token=22HWWFWPW5)](https://codecov.io/gh/childmindresearch/niftyone)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/childmindresearch/niftyone/blob/main/LICENSE)
<!-- [![Documentation](https://img.shields.io/badge/documentation-8CA1AF?logo=readthedocs&logoColor=fff)](https://childmindresearch.github.io/niftyone) -->

![NiftyOne Mosaic](.github/static/niftyone_mosaic_view.png)
</div>

NiftyOne is a platform for bulk visualization of large-scale neuroimaging datasets,
leveraging features of [FiftyOne] with popular neuroimaging python packages.

## Installation

> [!TIP]
> For stability, NiftyOne should be installed in its own environment. For example, to
> install NiftyOne using `conda`:
>
> ```sh
> conda create -y -n niftyone python=3.10
> conda activate niftyone
> ```

NiftyOne can be installed using pip:

```sh
pip install -U pip
pip install git+https://github.com/childmindresearch/niftyone.git
```

> [!IMPORTANT]
> [FFmpeg] is a non-Python dependency required for NiftyOne.
> Please refer to their documentation for installation instructions.

## Usage

To get started, try using the boilerplate command:

```sh
niftyone <bids_directory> <output_directory> <analysis-level>
```

> [!TIP]
> To see all arguments, run:
>
> ```sh
> niftyone --help
> ```

### Quick-start

1. Generate figures for each participant

    ```sh
    niftyone <bids_directory> <output_directory> participant
    ```

2. Collect participant figures into a compatible dataset

    ```sh
    niftyone <bids_directory> <output_directory> group
    ```

3. Launch FiftyOne app

    ```sh
    niftyone <bids_directory> <output_directory> launch
    ```

<!-- ## Documentation

For detailed information, including advanced usage, please visit our [documentation]. -->

<!-- ## Contributing

Contributions to NiftyOne are welcome! Please refer to the
[Contributions] page for information on how to contribute, report issues, or submit
pull requests. -->

## License

NiftyOne is distributed under the [MIT license].

## Support

If you encounter any issues or have questions, please open an issue on the
[issue tracker].

<!-- Links -->
[FiftyOne]: https://docs.voxel51.com/
[FFmpeg]: https://ffmpeg.org/
[MIT license]: https://github.com/childmindresearch/niftyone/blob/main/LICENSE
[issue tracker]: https://github.com/childmindresearch/niftyone/issues
