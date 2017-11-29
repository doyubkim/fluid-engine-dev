FROM ubuntu:14.04
MAINTAINER Doyub Kim <doyubkim@gmail.com>

RUN apt-get update -yq && \
    apt-get install -yq build-essential python-dev python-pip cmake

ADD . /app

WORKDIR /app/build
RUN cmake .. -DUSE_GL=OFF && make install

RUN apt-get install -yq pkg-config libfreetype6-dev libpng-dev
RUN pip install -r ../requirements.txt && pip install ..

WORKDIR /
