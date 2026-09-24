FROM python:3.11-slim

WORKDIR /app
COPY requirements-local.txt .
RUN pip install --no-cache-dir -r requirements-local.txt

COPY . .

# Build the index at container start if it is missing, then serve the UI.
EXPOSE 8501
CMD ["sh", "-c", "test -d data/index || python -m src.ingest; streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
