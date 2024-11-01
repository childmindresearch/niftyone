# Contributing

If you are looking to contribute, we recommend cloning the repository locally:

```sh
git clone https://github.com/childmindresearch/niftyone <local_directory>
cd <local_directory>    # Navigate to local copy of repository
```

## Dependencies

`niftyone` relies on a number of internal (python) and external (system) dependencies in
order to build a platform for bulk visualization of large-scale neuroimaging datasets.
The primary external dependency is [FFmpeg] - installation instructions can be found on
their website. Internal dependencies are listed in the [pyproject.toml] file of the
repository, with specific development versions listed in the [requirements.txt] file -
any version updates will also be reflected in this file.

## Development environment setup

To setup the development environment, first ensure external dependencies are installed.

> [!TIP]
> It is recommended to setup a virtual environment to install internal dependencies and
> for stability.
>
> ```sh
> python -m venv <venv_directory>
> source activate <venv_directory>/bin/activate
> ```

Internal dependencies can be installed using pip:

```sh
pip install -e .\[doc,dev,test\]
```

To ensure the installation was successful, try running the `niftyone` command:

```sh
niftyone -h
```

## Code formatting

`niftyone` uses `pre-commit`, as well as a Github action workflow to check for and address formatting issues.
These use the following:

- `ruff` - formatting and linting
- `mypy` - type checking
- `language-formatters-pre-commit-hooks` - pretty format YAML and TOML files
- `pre-commit-hooks` - fix string casing, format JSON files

To install the `pre-commit` configuration, run the following:

```sh
pre-commit install
```

## Adding features / fixing bugs

To contribute a change to the code base, checkout a new branch from the main branch and then make your changes.

```bash
git checkout -b feature/your-feature-name main
```

## Pull requests

Once you have made your changes and are ready to contribute, follow the steps to submit a pull request:

1. Push your changes back.

    ```bash
    git push -u origin feature/your-feature-name
    ```

1. Create a pull request to merge your branch into the main branch. Provide a clear
description of your changes in the pull request message.

### Guidelines

- Write clear and concise commit messages.
- Test your changes thoroughly before submitting a pull request
- If the pull request adds functionality, the documentation should also be updated.

> [!IMPORTANT]
> Contributed code will be **licensed under the same [license](LICENSE) as the rest of
> the repository**. If you did not write the code yourself, you must ensure the existing
> license is compatible and include the license information in the contributed files,
> or obtain permission from the original author to relicense the contributed code.

It is okay to submit work-in-progress and seek feedback - you will likely be asked to
make additional changes or asked clarification questions.

### Review process

All pull requests will undergo a review process before being accepted. Reviewers may
provide feedback or request changes to ensure the quality of the codebase.

<!-- Links -->
[FFmpeg]: https://ffmpeg.org/
[pyproject.toml]: https://github.com/childmindresearch/niftyone/blob/main/pyproject.toml
[requirements.txt]: https://github.com/childmindresearch/niftyone/blob/main/requirements.txt
