FROM osgeo/gdal:ubuntu-small-3.4.1 

# Labels
LABEL org.label-schema.schema-version="1.0"
LABEL org.label-schema.name="gingtran/geotools"
LABEL org.label-schema.description="GEO Data Sciences"
LABEL org.label-schema.version="1.0.0"
LABEL org.label-schema.vendor="Giang Tran"

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update \
    && apt install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install ffmpeg libsm6 libxext6  -y \
    && apt -y install python3.8 curl \
    && apt-get install -y python3-pip \
    && apt-get install -y gdal-bin \
    && apt-get install -y	python3-gdal
	
WORKDIR /src
ADD requirements.txt requirements.txt
RUN pip3 install --upgrade pip setuptools && \
    pip3 install -r requirements.txt 

ADD ./ /src
RUN python run.py

ENTRYPOINT ["src"]
