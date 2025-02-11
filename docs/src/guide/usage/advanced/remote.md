# Remote

NiftyOne can also be launched remotely (e.g. on a cluster). To use NiftyOne remotely,
first follow either the [local](../local.md) or [container](../container.md) usage
instructions with your remote server instructions (e.g. compute node if required). Once
NiftyOne is launched, a (multi-layered) port forward will need to be set up from your
local machine to the remote server:

## Single-layer port forwarding

The follow provides an example of a single-layer port forward:

```bash
HOST=hostname # Name of the remote server (not the full address)
PORT=5151 # Assume port 5151 is available, change this as needed

ssh -L ${PORT}:localhost:${PORT} ${USER}@remote.server.com
```

> [!NOTE]
> * `-L ${PORT}:localhost:${PORT}` maps the variable `$PORT` on your local machine
> to the same port on the remote server `localhost` (relative to `remote.server.com`)

From here, point the web browser on the local machine (e.g. your computer) to
`localhost:${PORT}`.

## Multi-layered port forwarding

The follow provides an example of a two-layer port forward (chain the command as needed
for more layers):

```bash
HOST=hostname # Name of the remote server (not the full address)
PORT=5151 # Assume port 5151 is available, change this as needed

ssh -L ${PORT}:localhost:${PORT} ${USER}@remote.server.com ssh -L ${PORT}:localhost:${PORT} -N $HOST
```

> [!NOTE]
> * `ssh -L ${PORT}:localhost:${PORT} ${USER}@remote.server.com` establishes an SSH
> connection to the intermediate server while forwarding the port
> * `ssh -L ${PORT}:localhost:${PORT} -N $HOST` forwards port from intermediate server
> to target remote host `$HOST`, with `-N` to keep the session open solely for port
> forwarding

From here, point the web browser on the local machine (e.g. your computer) to
`localhost:${PORT}` as before.
