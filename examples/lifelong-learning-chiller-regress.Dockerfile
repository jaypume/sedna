FROM tensorflow/tensorflow:1.15.4

RUN apt update \
  && apt install -y libgl1-mesa-glx
COPY ./lib/requirements.txt /home
# install requirements of sedna lib
RUN pip install -r /home/requirements.txt
RUN pip install joblib~=1.0.1
RUN pip install pandas~=1.1.5
RUN pip install scikit-learn~=0.24.1
RUN pip install xgboost~=1.3.3

ENV PYTHONPATH "/home/lib"

WORKDIR /home/work
COPY ./lib /home/lib

COPY examples/lifelong_learning/chiller  /home/work/


ENTRYPOINT ["python"]