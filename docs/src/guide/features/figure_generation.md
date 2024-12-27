# Figure Generation

<!-- Include some example images -->

NiftyOne can generate figures and videos for [BIDS] datasets for QC purposes using popular Python libraries, such as `nibabel`, `numpy`, etc. Possible outputs include:

## General / anatomical
* Three view - a static image for a given volume in the three anatomical orientations (coronal, axial, sagittal)
* Slice video - a video slicing through a single orientation for a given volume
* Three view video - a video slicing through the 4th dimension (e.g. time) in the three anatomical orientations

## Diffusion
* QSpace shells - visualization of diffusion gradients in Q-space
* DWI per shell - video visualizing diffusion volumes by gradient strength
* Signal per volume - 2-panel video, visualizing diffusion MRI volume with associated average signal per volume

## Functional
* Carpet plot - Visualization of carpet plot for functional data
* Mean / std - Visualization of mean and standard deviations of functional data

Overlaying of multiple volumes is also possible (e.g. to visualize masks, registration, etc.)

> [!NOTE]
> These are only some of the possible visualizations. Additional visualizations can be generated - see [advanced usage] for more details.

---
[BIDS]: https://bids-specification.readthedocs.io/en/stable/
[advanced usage]: #
