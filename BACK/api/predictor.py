"""
Clase para realizar predicciones con el modelo cargado
"""
import torch
import json
import os
from transformers import BertTokenizer, BertForSequenceClassification
from models.classifier import MovieSentimentClassifier
from data.preprocessing import clean_text
import config


class SentimentPredictor:
    """Predictor singleton para la API"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SentimentPredictor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.device = config.DEVICE
        self.max_length = config.MAX_LENGTH
        self.bert_model = None
        self.classifier_model = None
        self.tokenizer = None
        self.metadata = None
        self._initialized = True
        
        print(f"Inicializando predictor en dispositivo: {self.device}")
    
    def load_models(self):
        """Carga los modelos BERT y el clasificador"""
        if self.bert_model is not None:
            print("Modelos ya cargados")
            return
        
        print(f"Cargando modelo BERT: {config.PRETRAINED_MODEL_NAME}...")
        self.tokenizer = BertTokenizer.from_pretrained(config.PRETRAINED_MODEL_NAME)
        self.bert_model = BertForSequenceClassification.from_pretrained(
            config.PRETRAINED_MODEL_NAME
        ).to(self.device)
        self.bert_model.eval()
        
        print(f"Cargando clasificador desde: {config.CLASSIFIER_MODEL_PATH}...")
        self.classifier_model = MovieSentimentClassifier(
            input_dim=config.INPUT_DIM,
            hidden_dim_1=config.HIDDEN_DIM_1,
            hidden_dim_2=config.HIDDEN_DIM_2,
            dropout_rate=config.DROPOUT_RATE
        ).to(self.device)
        
        if not os.path.exists(config.CLASSIFIER_MODEL_PATH):
            raise FileNotFoundError(
                f"No se encontró el modelo entrenado en {config.CLASSIFIER_MODEL_PATH}. "
                "Por favor, ejecuta main.py primero para entrenar el modelo."
            )
        
        self.classifier_model.load_state_dict(
            torch.load(config.CLASSIFIER_MODEL_PATH, map_location=self.device)
        )
        self.classifier_model.eval()
        
        # Cargar metadata si existe
        if os.path.exists(config.MODEL_METADATA_PATH):
            with open(config.MODEL_METADATA_PATH, 'r') as f:
                self.metadata = json.load(f)
        
        print("Modelos cargados exitosamente")
    
    def is_loaded(self):
        """Verifica si los modelos están cargados"""
        return self.bert_model is not None and self.classifier_model is not None
    
    def predict(self, review_text: str, preprocess: bool = True):
        """
        Predice el sentimiento de una reseña
        
        Args:
            review_text: Texto de la reseña
            preprocess: Si aplicar preprocesamiento al texto
            
        Returns:
            dict con sentiment, confidence y probabilidades
        """
        if not self.is_loaded():
            raise RuntimeError("Los modelos no están cargados. Llama a load_models() primero.")
        
        # Preprocesar si es necesario
        if preprocess:
            processed_text = clean_text(review_text)
        else:
            processed_text = review_text
        
        # Tokenizar
        inputs = self.tokenizer(
            processed_text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Obtener embedding CLS desde BERT
            outputs = self.bert_model(**inputs, output_hidden_states=True)
            cls_embedding = outputs.hidden_states[-1][:, 0, :]
            cls_embedding = cls_embedding.to(next(self.classifier_model.parameters()).dtype)
            
            # Pasar por el clasificador
            logits = self.classifier_model(cls_embedding)
            prob_positive = torch.sigmoid(logits).item()
            prob_negative = 1 - prob_positive
            
            prediction = 1 if prob_positive > config.CLASSIFICATION_THRESHOLD else 0
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = prob_positive if prediction == 1 else prob_negative
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "probability_positive": prob_positive,
            "probability_negative": prob_negative
        }
    
    def predict_batch(self, reviews: list[str], preprocess: bool = True):
        """
        Predice el sentimiento de múltiples reseñas
        
        Args:
            reviews: Lista de textos de reseñas
            preprocess: Si aplicar preprocesamiento
            
        Returns:
            Lista de diccionarios con predicciones
        """
        results = []
        for review in reviews:
            try:
                prediction = self.predict(review, preprocess)
                results.append(prediction)
            except Exception as e:
                results.append({
                    "sentiment": "Error",
                    "confidence": 0.0,
                    "probability_positive": 0.0,
                    "probability_negative": 0.0,
                    "error": str(e)
                })
        return results
    
    def get_model_info(self):
        """Retorna información sobre el modelo"""
        info = {
            "model_name": config.PRETRAINED_MODEL_NAME,
            "classifier_architecture": {
                "input_dim": config.INPUT_DIM,
                "hidden_dim_1": config.HIDDEN_DIM_1,
                "hidden_dim_2": config.HIDDEN_DIM_2,
                "dropout_rate": config.DROPOUT_RATE
            },
            "device": self.device,
            "max_length": self.max_length
        }
        
        if self.metadata:
            info.update(self.metadata)
        
        return info
