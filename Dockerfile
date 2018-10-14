FROM ubuntu:14.04
MAINTAINER Doyub Kim <doyubkim@gmail.com>

RUN apt-get update -yq && \
    apt-get install -yq build-essential python-dev cmake curl

ADD . /app

WORKDIR /app/build
RUN cmake .. && \
    make -j`nproc` && \
    make install

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py
RUN apt-get install -yq pkg-config libfreetype6-dev libpng-dev
RUN pip install -r ../requirements.txt && \
    pip install ..

WORKDIR /
