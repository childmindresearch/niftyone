
# NiftyOne

![NiftyOne Mosaic](.github/static/niftyone_mosaic_view.png)

NiftyOne is a platform for bulk visualization of large-scale neuroimaging datasets. It is built with [FiftyOne](https://docs.voxel51.com/).

## Installation
**1. Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)**
For stability, NiftyOne should be installed in it's own environment. For example,

```bash
conda create -y -n niftyone python=3.10
conda activate niftyone

pip install -U pip
pip install git+https://github.com/cmi-dair/niftyone.git
```
## Testing
1. Install [datalad](https://www.datalad.org) to download the data files
2. You can use whatever BIDS dataset (such as [ds000102](https://openneuro.org/datasets/ds000102/versions/00001)  from OpenNeuro) as an example to test 
```bash
niftyone bids_dir output_dir participant --workers 8
```
## Usage
### 1. Test launching the app
```bash
niftyone app launch
```

### 2. Generate figures for each participant

```bash
niftyone bids_dir output_dir participant --workers 8
```

### 3. Collect participant figures into a FiftyOne dataset

```bash
niftyone bids_dir output_dir group
```

### 4. Launch FiftyOne app

```bash
niftyone bids_dir output_dir launch
```
## Running NiftyOne on a remote cluster
### 1) SSH into the login node
from a Mac/Linux/Windows Powershell terminal:
```bash
ssh user@server.com
```
  

### 2) switch from the login node to a compute node
```
e.g. interact
```
  
### 3) Follow steps above to create a conda environment, install, and launch the app

  
At this point you should see "App Launched" in your terminal with a localhost URL.

 
### 4) Port Forwarding

Open a second terminal.

Execute the following command to create a multihop port forward from your local machine --> login node --> compute node. Related [gist](https://gist.github.com/clane9/ea3a469e727b6e7f75dc4373e9d2241d)

  
HOST=(name of compute node, not the full address. e.g. r001)
PORT=5151
```
ssh -L $PORT:localhost:$PORT $USER@server.com ssh -L $PORT:localhost:$PORT -N $HOST
```
  

### 5) Open [http://localhost:5151](http://localhost:5151) on your local browser to check the app.

  

### Additional reference: 

https://superuser.com/questions/1154383/ssh-port-forwarding-on-windows

https://phoenixnap.com/kb/ssh-port-forwarding
## FAQ

### Q1) Where can NiftyOne be installed?

NiftyOne can be installed both locally and on a remote server. If NiftyOne is installed on a server, the output can be viewed from a local web browser after port fowarding.

  

### Q2) Port forwarding is not working and I am getting connection errors

Try testing with a simple HTTP server with a "Hello World" HTML file using port fowarding to rule out any networking problems before installing further software.

  
$python3 -m http.server 9000

https://www.digitalocean.com/community/tutorials/python-simplehttpserver-http-server#python-simplehttpserver-error-no-module-named-simplehttpserver

  
 
###  Q3) I am getting the following error:

/home/user/.../mongod: error while loading shared libraries

  

Try re-installing mongo and re-creating the virtual conda environment.

  

###  Q4) I am getting the following error:

OSError: You must have fiftyone>=0.22.1 installed in order to migrate from v0.22.1 to v0.21.6, but you are currently running fiftyone==0.21.6.

  

Try re-installing fiftyone / niftyone

  

###  Q5) I am getting an error about connection failure to port 5151

Try incrementing to port 5152, 5153, etc
