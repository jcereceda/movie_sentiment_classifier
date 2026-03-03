"""
Esquemas de datos para la API usando Pydantic
"""
from pydantic import BaseModel, Field
from typing import Optional


class ReviewRequest(BaseModel):
    """Esquema para la solicitud de predicción"""
    review: str = Field(
        ..., 
        min_length=1,
        max_length=5000,
        description="Texto de la reseña de película a analizar",
        example="This movie was absolutely fantastic! The acting was superb."
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "review": "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout."
            }
        }


class SentimentResponse(BaseModel):
    """Esquema para la respuesta de predicción"""
    review: str = Field(..., description="Texto de la reseña analizada")
    sentiment: str = Field(..., description="Sentimiento predicho: 'Positive' o 'Negative'")
    confidence: float = Field(..., description="Nivel de confianza de la predicción (0-1)")
    probability_positive: float = Field(..., description="Probabilidad de ser positivo")
    probability_negative: float = Field(..., description="Probabilidad de ser negativo")
    
    class Config:
        json_schema_extra = {
            "example": {
                "review": "This movie was absolutely fantastic!",
                "sentiment": "Positive",
                "confidence": 0.9523,
                "probability_positive": 0.9523,
                "probability_negative": 0.0477
            }
        }


class BatchReviewRequest(BaseModel):
    """Esquema para predicción en lote"""
    reviews: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Lista de reseñas a analizar (máximo 100)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "reviews": [
                    "Great movie, loved every minute!",
                    "Terrible film, waste of time.",
                    "It was okay, nothing special."
                ]
            }
        }


class BatchSentimentResponse(BaseModel):
    """Esquema para respuesta de predicción en lote"""
    results: list[SentimentResponse]
    total_processed: int = Field(..., description="Número total de reseñas procesadas")


class HealthResponse(BaseModel):
    """Esquema para el endpoint de salud"""
    status: str = Field(..., description="Estado del servicio")
    model_loaded: bool = Field(..., description="Si el modelo está cargado")
    device: str = Field(..., description="Dispositivo utilizado (cuda/cpu)")
    version: str = Field(..., description="Versión de la API")


class ModelInfoResponse(BaseModel):
    """Esquema para información del modelo"""
    model_name: str = Field(..., description="Nombre del modelo BERT utilizado")
    classifier_architecture: dict = Field(..., description="Arquitectura del clasificador")
    training_date: Optional[str] = Field(None, description="Fecha de entrenamiento")
    accuracy: Optional[float] = Field(None, description="Accuracy en validación")
    f2_score: Optional[float] = Field(None, description="F2-Score en validación")

class ReviewInsertRequest(BaseModel):
    """Esquema para inserar nuevas reseñas"""
    review: str = Field(..., min_length=1)
    sentiment: str = Field(..., pattern="^(positive|negative)$")
    
    class Config:
        json_schema_extra = {
            "example": {
                "review": "This was a really good movie",
                "sentiment": "positive"
            }
        }


class ReviewInsertResponse(BaseModel):
    """Respuesta de inserción de reseña"""
    message: str
    review_id: str
    review: str
    sentiment: str

class DatabaseStatsResponse(BaseModel):
    """Estadisticas de la base de datos"""
    total_reviews: int
    positive_reviews: int
    negative_reviews: int
    positive_percentage: float
    negative_percentage: float