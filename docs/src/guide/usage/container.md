# Container

Using NiftyOne via a container (e.g. Docker, Singularity), is
similar to using NiftyOne [locally](./local.md). There are a few
additional flags required.

NiftyOne uses a hidden `.fiftyone` directory stored in the home directory
(e.g. `/home/user/.fiftyone`) This directory needs to be writable
and mounted with the appropriate container flag.

## Docker

First download the container, replacing `<TAG>` in the following
command with the version to be pulled:

```bash
docker pull ghcr.io/childmindresearch/niftyone:<TAG>
```

To use NiftyOne with Docker, the host network also needs to be
made available to be able to pass the launched application.
Rather than executing each workflow with individual Docker
commands, it is recommended to shell into the container
and call each command:

```bash
docker run -it \
  -v "/home/user/.fiftyone:/.fiftyone:rw" \
  --network host \
  childmindresearch/niftyone:<TAG>
```

Once you are shelled into the container, you can follow the same
instructions as if you are running NiftyOne [locally](./local.md).
