# Metrics

NiftyOne can also aggregate QC metrics associated with each participant.

<!-- Include example showing QC metric on niftyone -->

> [!NOTE]
> Currently, NiftyOne expects this to follow similarly to MRIQC group outputs. That is, a single tab-separated file with the following file name pattern: `group_<suffix>.tsv`, where `<suffix>` is associated with the BIDS entity (e.g. `T1w`). Each column is a separate QC metric, and each row is an individual participant.

For an example QC file, see the included QC metrics in the [test dataset](https://github.com/OpenNeuroDerivatives/ds000102-mriqc/blob/cd0559d460a794e553f3f42e1f09ff063069dfa2/group_T1w.tsv).


---
[MRIQC]: https://mriqc.readthedocs.io/en/latest/index.html
