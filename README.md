# NiftyOne

[![Build](https://github.com/childmindresearch/niftyone/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/childmindresearch/niftyone/actions/workflows/test.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/childmindresearch/niftyone/branch/main/graph/badge.svg?token=22HWWFWPW5)](https://codecov.io/gh/childmindresearch/niftyone)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![stability-orange](https://img.shields.io/badge/stability-experimental-orange.svg)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/childmindresearch/niftyone/blob/main/LICENSE)
[![pages](https://img.shields.io/badge/api-docs-blue)](https://childmindresearch.github.io/niftyone)

![NiftyOne Mosaic](.github/static/niftyone_mosaic_view.png)

NiftyOne is a platform for bulk visualization of large-scale neuroimaging datasets, built upon [FiftyOne](https://docs.voxel51.com/).

## Installation

For stability, NiftyOne is recommended to be installed in it's own environment.

```bash
python -m venv niftyone-venv
source niftyone-venv/bin/activate

pip install -U pip
pip install git+https://github.com/childmindresearch/niftyone
```

## Quick start

Below are some commands to help you quickly get started.

### 1. Generate participant figures

```bash
niftyone bids_dir output_dir participant
```

### 2. Collect participant figures into a FiftyOne dataset

```bash
niftyone bids_dir output_dir group
```

### 3. Launch FiftyOne app

```bash
niftyone bids_dir output_dir launch
```
