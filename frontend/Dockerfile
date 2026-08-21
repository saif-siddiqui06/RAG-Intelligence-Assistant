FROM python:3.11-slim

WORKDIR /app

# The frontend only imports streamlit + requests (see api_client.py) —
# installed directly rather than from requirements.txt, which also pulls
# in the backend's torch/faiss/sentence-transformers stack this image
# never uses.
RUN pip install --no-cache-dir "streamlit>=1.39,<2.0" "requests>=2.32,<3.0"

COPY frontend ./frontend

ENV PYTHONUNBUFFERED=1

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"

CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
