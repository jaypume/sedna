FROM tensorflow/tensorflow:1.15.4

RUN apt update \
  && apt install -y libgl1-mesa-glx

COPY ./lib/requirements.txt /home

RUN pip install -r /home/requirements.txt

ENV PYTHONPATH "/home/lib:/home/plato"

COPY ./lib /home/lib
COPY ./plato /home/plato

WORKDIR /home/work
COPY examples/federated_learning/surface_defect_detection/aggregation_worker/  /home/work/

ENTRYPOINT ["python", "aggregate.py"]
