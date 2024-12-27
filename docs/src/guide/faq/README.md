# Frequently Asked Questions (FAQ)

1. **Where can NiftyOne be installed?**

    NiftyOne can be installed both locally and on a remote serve. If NiftyOne is installed on a server, the output can be viewed from a local web browser after port forwarding.

2. **I am getting the following error: `/home/user/.../mongod: error while loading shared libraries`.**

    Try to reinstall `mongo` and / or re-creating the virtual environment `NiftyOne` is installed in.

3. **I am getting the following error: `OSError: You must have fiftyone>=0.22.1 installed in order to migrate from v0.22.1 to v0.21.6, but you are currently running fiftyone==0.21.6`.**

    Try re-installing `fiftyone` / `niftyone`.

4. **I am getting an error about connection failure to port 5151.**

    The port 5151 is likely already being used by another application - try incrementing the port number to 5152, 5153, etc.
