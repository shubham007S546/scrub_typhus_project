FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train model on first build
RUN python ml/train_model.py

EXPOSE 8000

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
