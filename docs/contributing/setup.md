## Setup development environment

This project requires Python 3.10 or higher and all python dependencies can be
found in the `pyproject.toml`. Additionally, NiftyOne requires a few
dependencies, including:

- `openssl`
- `libcurl` (depending on Linux distribution, `libcurl3` or `libcurl4` may need to be specified)
- `ffmpeg` (for video datasets)

It is also recommended to clone the repository and create a virtual
environment for development.

```bash
# Clone repository
git clone https://github.com/childmindresearch/niftyone
cd niftyone

# Install dependencies
pip install -U pip
pip install -e .\[dev,test,docs\]
```

### Dependencies

The `requirements.txt` file provides pinned versions of the core dependencies.
If any new package is required or versions are updated, please also update
this file.
