# Frequently asked questions (FAQ)

### 1. Where can NiftyOne be installed?

NiftyOne can be installed both locally and on a remote server. If NiftyOne is
installed on a server, the output can be viewed from a local web browser after
the ports have been forwarded.

### 2. I am getting the following follow:

```bash
/home/user/.../mongod: error while loading shared libraries
```

Ensure all necessary and correct library dependencies are installed.

### 3. I am getting a similar error to the following:

```bash
OSError: You must have fiftyone>=0.22.1 installed in order to migrate
from v0.22.1 to v0.21.6, but you are currently running fiftyone==0.21.6.
```

Try updating or re-installing `fiftyone` / `niftyone`!
