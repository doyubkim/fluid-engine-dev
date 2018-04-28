#!/usr/bin/env bash

set -e

if [ $# -eq 0 ]
  then
    docker build -t doyubkim/fluid-engine-dev .
else
    docker build -f $1 -t doyubkim/fluid-engine-dev:$2 .
fi
docker run doyubkim/fluid-engine-dev
