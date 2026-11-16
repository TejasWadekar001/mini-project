FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
# Needs a trained model at runs/deploy/best.keras and runs/splits.json
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
