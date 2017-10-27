FROM ubuntu:14.04
MAINTAINER Doyub Kim <doyubkim@gmail.com>

RUN apt-get update -yq && \
    apt-get install -yq build-essential python-dev python-pip cmake libglfw3-dev

ADD . /app

WORKDIR /app/build
RUN cmake .. && \
    make -j`nproc` && \
    make install

RUN apt-get install -yq pkg-config libfreetype6-dev libpng-dev
RUN pip install -r ../requirements.txt && \
    pip install ..

WORKDIR /
