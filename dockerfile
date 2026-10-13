FROM python:3.11-slim

WORKDIR /app
ENV HF_HOME=/app/hf_cache

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY main.py .


RUN mkdir -p /app/cache_data && chmod -R 777 /app/cache_data /app/hf_cache

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]