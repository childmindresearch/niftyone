FROM ubuntu:20.04

RUN apt-get update && apt-get install -y python3-dev python3-pip git build-essential

ADD requirements.txt /opt/requirements.txt

RUN pip install -r https://raw.githubusercontent.com/cmi-dair/niftyone/main/requirements.txt
RUN pip install git+https://github.com/cmi-dair/niftyone.git

EXPOSE 5151
ENTRYPOINT ["niftyone"]
