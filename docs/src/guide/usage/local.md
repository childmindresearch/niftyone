# Local

To use NiftyOne locally, first follow installation
[instructions](../installation/README.md). Once installed, NiftyOne can be initiated
from the terminal.

> [!NOTE]
> Pre-computing a bids2table index path for a dataset is
> not required, but recommended.

## 1. Generating figures

We'll start by generating figures using the `participant` level workflow:

```bash
niftyone bids_dir output_dir participant --verbose
```

## 2. Aggregating data

Once figures are generated, we can aggregate figures (and QC metrics
if available). For this example, we'll assume there are only figures.
Additionally, we'll give the dataset a unique name - if one is not
provided, it will assume this from the bids directory.

```bash
niftyone bids_dir output_dir group --ds-name example --verbose
```

## Launching applicaiton

Finally, we can launch our application with the dataset by including
the dataset name we used in the previous step.

```bash
niftyone bids_dir output_dir launch --ds-name example
```

If successfully launched, a message should appear in the terminal:

```bash
App launched. Point your web browser to http://localhost:5151

To exit, close the App or press ctrl + c
```

Enter the address in your web browser or choice to launch the application.

<!-- Add screenshot -->
