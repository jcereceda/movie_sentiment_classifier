"""
API REST para clasificación de sentimientos en reseñas de películas
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager
from data.database import MongoDBHandler

from api.schemas import (
    ReviewRequest, 
    SentimentResponse, 
    BatchReviewRequest,
    BatchSentimentResponse,
    HealthResponse,
    ModelInfoResponse,
    ReviewInsertRequest,
    ReviewInsertResponse,
    DatabaseStatsResponse
)
from api.predictor import SentimentPredictor
import config


# Inicializar predictor global
predictor = SentimentPredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación"""
    # Startup: Cargar modelos
    print("Iniciando API...")
    try:
        predictor.load_models()
        print("API lista para recibir peticiones")
    except Exception as e:
        print(f"Error al cargar modelos: {e}")
        raise
    
    yield
    
    # Shutdown
    print("Cerrando API...")


# Crear aplicación FastAPI
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION,
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    """Endpoint raíz con información básica"""
    return {
        "message": "Movie Review Sentiment Analysis API",
        "version": config.API_VERSION,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Verifica el estado de salud del servicio"""
    return HealthResponse(
        status="healthy" if predictor.is_loaded() else "unhealthy",
        model_loaded=predictor.is_loaded(),
        device=config.DEVICE,
        version=config.API_VERSION
    )


@app.post("/predict", response_model=SentimentResponse, tags=["Prediction"])
async def predict_sentiment(request: ReviewRequest):
    """
    Predice el sentimiento de una reseña de película
    
    - **review**: Texto de la reseña a analizar
    
    Retorna el sentimiento (Positive/Negative) con nivel de confianza
    """
    try:
        if not predictor.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="El modelo no está cargado. Intenta nuevamente en unos momentos."
            )
        
        # Realizar predicción
        prediction = predictor.predict(request.review, preprocess=True)
        
        return SentimentResponse(
            review=request.review,
            sentiment=prediction["sentiment"],
            confidence=prediction["confidence"],
            probability_positive=prediction["probability_positive"],
            probability_negative=prediction["probability_negative"]
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al procesar la predicción: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchSentimentResponse, tags=["Prediction"])
async def predict_batch(request: BatchReviewRequest):
    """
    Predice el sentimiento de múltiples reseñas en lote
    
    - **reviews**: Lista de reseñas a analizar (máximo 100)
    
    Retorna una lista con las predicciones para cada reseña
    """
    try:
        if not predictor.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="El modelo no está cargado. Intenta nuevamente en unos momentos."
            )
        
        if len(request.reviews) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Máximo 100 reseñas por solicitud"
            )
        
        # Realizar predicciones
        predictions = predictor.predict_batch(request.reviews, preprocess=True)
        
        results = [
            SentimentResponse(
                review=review,
                sentiment=pred["sentiment"],
                confidence=pred["confidence"],
                probability_positive=pred["probability_positive"],
                probability_negative=pred["probability_negative"]
            )
            for review, pred in zip(request.reviews, predictions)
        ]
        
        return BatchSentimentResponse(
            results=results,
            total_processed=len(results)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al procesar las predicciones: {str(e)}"
        )

@app.get("/database/stats", response_model=DatabaseStatsResponse, tags=["Database"])
async def get_database_stats():
    """
    Obtiene estadisticas de la BD MongoDB
    
    Retorna la cantidad de reseñas y su distribución por sentimiento
    """
    try:
        db_handler = MongoDBHandler()
        counts = db_handler.count_reviews()
        db_handler.close()

        total = counts['total']
        positive = counts['positive']
        negative = counts['negative']

        return DatabaseStatsResponse(
            total_reviews=total,
            positive_reviews=positive,
            negative_reviews=negative,
            positive_percentage=round((positive / total * 100) if total > 0 else 0, 2),
            negative_percentage=round((negative / total * 100) if total > 0 else 0, 2)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener estadisticas: {str(e)}"
        )
    

@app.post("/database/review", response_model=ReviewInsertResponse, tags=["Database"])
async def insert_review(request: ReviewInsertRequest):  
    """
    Inserta nueva reseña en la base de datos

    -**review**: Texto de la reseña
    -**sentiment**: Sentimiento positive o negative
    
    """
    try:
        from data.preprocessing import clean_text

        db_handler = MongoDBHandler()

        sentiment_val = 1 if request.sentiment == "positive" else 0
        clean_review = clean_text(request.review)

        review_id = db_handler.insert_review(
            review=request.review,
            sentiment=request.sentiment,
            setiment_val=sentiment_val,
            clean_review=clean_review
        )

        db_handler.close()

        return ReviewInsertResponse(
            message="Reseña insertada correctamente",
            review_id=review_id,
            review=request.review,
            sentiment=request.sentiment
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al insertar reseñas: {str(e)}"
        )
    



@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Obtiene información sobre el modelo cargado
    
    Retorna detalles sobre la arquitectura, métricas de entrenamiento, etc.
    """
    try:
        if not predictor.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="El modelo no está cargado"
            )
        
        info = predictor.get_model_info()
        return ModelInfoResponse(**info)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener información del modelo: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Manejador global de excepciones"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Error interno del servidor",
            "error": str(exc)
        }
    )


def start_api():
    """Inicia el servidor API"""
    uvicorn.run(
        "app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,  # Cambiar a True en desarrollo
        log_level="info"
    )


if __name__ == "__main__":
    start_api()
