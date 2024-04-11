# NiftyOne

![NiftyOne Mosaic](.github/static/niftyone_mosaic_view.png)

NiftyOne is a platform for bulk visualization of large-scale neuroimaging datasets. It is built with [FiftyOne](https://docs.voxel51.com/).

## Installation

For stability, NiftyOne should be installed in it's own environment. For example, to install using python 3.10:

```bash
conda create -y -n niftyone python=3.10
conda activate niftyone

pip install -U pip
pip install git+https://github.com/childmindresearch/niftyone.git
```

## Usage

### 1. Generate figures for each participant

```bash
niftyone bids_dir output_dir participant --workers 8
```

### 2. Collect participant figures into a FiftyOne dataset

```bash
niftyone bids_dir output_dir group
```

### 3. Launch FiftyOne app

```bash
niftyone bids_dir output_dir launch
```
