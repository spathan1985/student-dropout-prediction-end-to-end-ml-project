FROM python:3.12

WORKDIR /code

# Copy requirements first for better caching
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create models directory
RUN mkdir -p /code/models

# Copy the model file
COPY ./models/student_dropout_model.pkl /code/models/

# Copy source code
COPY ./src /code/src

# Update model path in environment variable
# ENV MODEL_PATH=/code/models/student_dropout_model.pkl

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "80"]