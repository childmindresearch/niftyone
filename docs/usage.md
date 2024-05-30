# Usage

## Command line interface (CLI)

The following can also be seen by running `niftyone -h` in your terminal.

Below are all arguments of the NiftyOne CLI, separated by analysis level.
In most cases, only the required arguments are needed.

```bash
usage: niftyone bids_dir output_dir analysis_level [options]

NiftyOne is a comphrensive tool designed to aid large-scale QC of BIDS
datasets through visualization and quantitative metrics.
```

### Positional arguments:
  `bids_dir`
  > Path to BIDS dataset

  `output_dir`
  > Path to output directory

  `analysis_level`
  > Analysis level

### Standard options:

`-h, --help`
> show this help message and exit

`--overwrite, -x`
> Overwrite previous results

`--verbose, -v`
> Verbose logging.


### Participant level options

`--participant-label LABEL, --sub LABEL`
> Participant to analyze (default: all)

`--index PATH`
> Pre-computed bids2table index path (default: {bids_dir}/index.b2t)

`--qc-dir PATH`
> Path to pre-computed QC outputs (default: {bids_dir}/derivatives/mriqc)

`--workers COUNT, -w COUNT`
>Number of worker processes. Setting to -1 runs as many processes as there
are cores available. (default: 1)


### Group level options:

`--ds-name DATASET`
> Name of NiftyOne dataset.

### Launch level options:

`--qc-key LABEL`
> Extra identifier for the QC session
