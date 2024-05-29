ARG PY_VERSION=3.11
FROM python:${PY_VERSION}-slim-bookworm as python
RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
    build-essential \
    ffmpeg \
    libcurl4 \
    git \
    openssl \
    && python -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# NOTE: Set version to be able to checkout tag in future
# (for now, installing latest from `main`)
FROM python as builder
ARG NIFTYONE_VER=0.0.0
COPY . /opt/niftyone/
RUN cd /opt/niftyone \
    && pip wheel --no-deps --prefer-binary .

FROM builder as runtime
WORKDIR /home
ENV OS=LINUX
RUN WHEEL=`ls /opt/niftyone | grep whl` \
    && pip install /opt/niftyone/${WHEEL} \
    && rm -r /opt/niftyone \
    && apt-get --purge -y -qq autoremove
EXPOSE 5151
ENTRYPOINT ["niftyone"]
