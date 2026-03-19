"""
Configuración centralizada del proyecto
"""
import torch
import os
from dotenv import load_dotenv

load_dotenv()

# Configuración general
SEED = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Rutas de datos y modelos
DATA_PATH = "IMDB_Dataset.csv"
MODEL_DIR = "saved_models"
CLASSIFIER_MODEL_PATH = os.path.join(MODEL_DIR, "classifier_model.pth")
MODEL_METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# Crear directorio de modelos si no existe
os.makedirs(MODEL_DIR, exist_ok=True)

# Configuración del modelo pre-entrenado BERT
PRETRAINED_MODEL_NAME = "textattack/bert-base-uncased-imdb"
MAX_LENGTH = 128

# Configuración de entrenamiento
BATCH_SIZE = 32
EMBEDDING_BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20
PATIENCE = 4
TRAIN_TEST_SPLIT = 0.2

# Configuración de la red neuronal
INPUT_DIM = 768  # Dimensión del embedding de BERT
HIDDEN_DIM_1 = 128
HIDDEN_DIM_2 = 64
DROPOUT_RATE = 0.3

# Umbral de clasificación
CLASSIFICATION_THRESHOLD = 0.5

# Configuración de la API
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Movie Review Sentiment Analysis API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
API para clasificación de sentimientos en reseñas de películas.

Utiliza un modelo BERT fine-tuned combinado con una red neuronal 
para determinar si una reseña es positiva o negativa.
"""

# Configuracion de MongoDB
MONGODB_URL = os.getenv("MONGODB_URL","mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE","sentiment_classiffication")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION","sentiment_classiffication")

# Configuración del Scheduler de Reentrenamiento
RETRAIN_ENABLED = os.getenv("RETRAIN_ENABLED", "true").lower() == "true"
RETRAIN_SCHEDULE = os.getenv("RETRAIN_SCHEDULE", "daily")  # daily, weekly, interval
RETRAIN_HOUR = int(os.getenv("RETRAIN_HOUR", "2"))  # Hora del día (0-23)
RETRAIN_MINUTE = int(os.getenv("RETRAIN_MINUTE", "0"))  # Minuto (0-59)
RETRAIN_DAY_OF_WEEK = os.getenv("RETRAIN_DAY_OF_WEEK", "mon")  # Para schedule weekly
RETRAIN_INTERVAL_HOURS = int(os.getenv("RETRAIN_INTERVAL_HOURS", "24"))  # Para schedule interval
RETRAIN_MIN_IMPROVEMENT = float(os.getenv("RETRAIN_MIN_IMPROVEMENT", "0.001"))  # Mejora mínima
