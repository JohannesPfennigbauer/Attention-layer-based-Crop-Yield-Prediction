FROM python:3.10-slim
WORKDIR /app

# Install dependencies 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run application
COPY ./app ./app
EXPOSE 8501
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]