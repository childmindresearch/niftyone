ARG FO_VERSION=0.25.0
ARG PY_VERSION=python3.11
FROM voxel51/fiftyone:${FO_VERSION}-${PY_VERSION} AS builder

# Will install version currently checked out
FROM builder AS niftyone
COPY . /opt/niftyone/
RUN apt-get update -qq \
    && apt-get install -y git \
    && cd /opt/niftyone \
    && pip wheel --no-deps --prefer-binary . --wheel wheels/

FROM builder AS runtime
COPY --from=niftyone /opt/niftyone/wheels/*.whl /opt/wheels/
WORKDIR /home
ARG ROOT_DIR=/home/.fiftyone
ENV OS=LINUX
RUN WHEEL=`ls /opt/wheels | grep whl` \
    && pip install /opt/wheels/${WHEEL} \
    && rm -r /opt/wheels
EXPOSE 5151
ENTRYPOINT ["niftyone"]
