FROM python:3.8.10-slim

RUN pip install pipenv

WORKDIR /app                                                                

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --system

COPY ["*.py", "kitchenware_model.tflite", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]