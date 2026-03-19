"""
API REST para clasificación de sentimientos en reseñas de películas
"""
import json
import os

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
    DatabaseStatsResponse,
    TrainingHistoryData,
    ConfusionMatrixData,
    ModelMetricsResponse
)
from api.predictor import SentimentPredictor
from training.scheduler import RetrainingScheduler
import config


# Inicializar predictor global
predictor = SentimentPredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación"""
    global scheduler
    
    # Startup: Cargar modelos e iniciar scheduler
    print("Iniciando API...")
    try:
        predictor.load_models()
        print("API lista para recibir peticiones")
        
        # Iniciar scheduler si está habilitado
        if config.RETRAIN_ENABLED:
            print("\n" + "="*80)
            print("CONFIGURANDO REENTRENAMIENTO AUTOMÁTICO")
            print("="*80)
            
            scheduler = RetrainingScheduler()
            
            # Configurar según el tipo de schedule
            if config.RETRAIN_SCHEDULE == "daily":
                scheduler.schedule_daily(
                    hour=config.RETRAIN_HOUR,
                    minute=config.RETRAIN_MINUTE,
                    min_improvement=config.RETRAIN_MIN_IMPROVEMENT
                )
            elif config.RETRAIN_SCHEDULE == "weekly":
                scheduler.schedule_weekly(
                    day_of_week=config.RETRAIN_DAY_OF_WEEK,
                    hour=config.RETRAIN_HOUR,
                    minute=config.RETRAIN_MINUTE,
                    min_improvement=config.RETRAIN_MIN_IMPROVEMENT
                )
            elif config.RETRAIN_SCHEDULE == "interval":
                scheduler.schedule_interval(
                    hours=config.RETRAIN_INTERVAL_HOURS,
                    min_improvement=config.RETRAIN_MIN_IMPROVEMENT
                )
            
            scheduler.start()
            
            next_run = scheduler.get_next_run_time()
            if next_run:
                print(f"⏰ Próximo reentrenamiento: {next_run}")
        else:
            print("\nℹ️  Reentrenamiento automático deshabilitado")
    
    except Exception as e:
        print(f"Error al cargar modelos: {e}")
        raise
    
    yield
    
    # Shutdown
    print("\nCerrando API...")
    if scheduler:
        scheduler.stop()
    print("API cerrada")


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
            sentiment_val=sentiment_val,
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

@app.post("/admin/retrain", tags=["Admin"])
async def trigger_manual_retrain(min_improvement: float = 0.001):
    """
    Ejecuta un reentrenamiento manual del modelo
    
    - **min_improvement**: Mejora mínima de accuracy requerida (default: 0.001)
    
    ⚠️ Este proceso puede tardar varios minutos
    """
    if not config.RETRAIN_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="El reentrenamiento automático está deshabilitado"
        )
    
    try:
        from training.auto_retrain import AutoRetrainer
        
        print("\n🔧 Reentrenamiento manual solicitado vía API")
        retrainer = AutoRetrainer()
        result = retrainer.run_auto_retrain(min_improvement=min_improvement)
        
        # Si el modelo se actualizó, recargar el predictor
        if result['model_updated']:
            print("🔄 Recargando modelo en el predictor...")
            predictor.load_models()
        
        return {
            "success": result['success'],
            "model_updated": result['model_updated'],
            "message": result['message'],
            "current_accuracy": result['current_accuracy'],
            "new_accuracy": result['new_accuracy'],
            "improvement": result['new_accuracy'] - result['current_accuracy']
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante el reentrenamiento: {str(e)}"
        )


@app.get("/admin/retrain/status", tags=["Admin"])
async def get_retrain_status():
    """
    Obtiene el estado del sistema de reentrenamiento automático
    
    Retorna información sobre la configuración y próxima ejecución
    """
    if not scheduler:
        return {
            "enabled": False,
            "message": "Reentrenamiento automático deshabilitado"
        }
    
    next_run = scheduler.get_next_run_time()
    
    return {
        "enabled": config.RETRAIN_ENABLED,
        "schedule_type": config.RETRAIN_SCHEDULE,
        "next_run": next_run.isoformat() if next_run else None,
        "min_improvement_threshold": config.RETRAIN_MIN_IMPROVEMENT,
        "configuration": {
            "hour": config.RETRAIN_HOUR if config.RETRAIN_SCHEDULE in ["daily", "weekly"] else None,
            "minute": config.RETRAIN_MINUTE if config.RETRAIN_SCHEDULE in ["daily", "weekly"] else None,
            "day_of_week": config.RETRAIN_DAY_OF_WEEK if config.RETRAIN_SCHEDULE == "weekly" else None,
            "interval_hours": config.RETRAIN_INTERVAL_HOURS if config.RETRAIN_SCHEDULE == "interval" else None
        }
    }


@app.get("/admin/retrain/history", tags=["Admin"])
async def get_retrain_history():
    """
    Obtiene el historial de modelos entrenados (backups)
    
    Retorna información sobre los backups disponibles
    """
    import os
    from pathlib import Path
    
    backup_dir = os.path.join(config.MODEL_DIR, "backups")
    
    if not os.path.exists(backup_dir):
        return {"backups": [], "total": 0}
    
    backups = []
    for backup_name in sorted(os.listdir(backup_dir), reverse=True):
        backup_path = os.path.join(backup_dir, backup_name)
        metadata_path = os.path.join(backup_path, "model_metadata.json")
        
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            backups.append({
                "name": backup_name,
                "date": metadata.get('training_date'),
                "accuracy": metadata.get('accuracy'),
                "f2_score": metadata.get('f2_score')
            })
    
    return {
        "backups": backups,
        "total": len(backups)
    }

@app.get("/model/metrics", response_model=ModelMetricsResponse, tags=["Model"])
async def get_model_metrics():
    """
    Obtiene las métricas completas del modelo actual
    
    Retorna:
    - Accuracy y F2-Score
    - Historial de entrenamiento (pérdidas y accuracies por época)
    - Matriz de confusión
    - Información adicional del entrenamiento
    
    Estos datos pueden ser usados para generar visualizaciones en el cliente
    """
    try:
        if not predictor.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="El modelo no está cargado"
            )
        
        # Cargar metadata
        if not os.path.exists(config.MODEL_METADATA_PATH):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No se encontró metadata del modelo. Entrena el modelo primero."
            )
        
        with open(config.MODEL_METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        
        # Extraer datos del historial de entrenamiento
        training_history_data = None
        if 'training_history' in metadata:
            hist = metadata['training_history']
            num_epochs = len(hist['train_losses'])
            
            training_history_data = TrainingHistoryData(
                epochs=list(range(1, num_epochs + 1)),
                train_losses=hist['train_losses'],
                val_losses=hist['val_losses'],
                val_accuracies=hist['val_accuracies'],
                best_epoch=hist['best_epoch'],
                best_val_loss=hist['best_val_loss']
            )
        
        # Extraer datos de la matriz de confusión
        cm = metadata['confusion_matrix']
        confusion_matrix_data = ConfusionMatrixData(
            true_negatives=cm['true_negatives'],
            false_positives=cm['false_positives'],
            false_negatives=cm['false_negatives'],
            true_positives=cm['true_positives'],
            matrix=[
                [cm['true_negatives'], cm['false_positives']],
                [cm['false_negatives'], cm['true_positives']]
            ]
        )
        
        # Construir respuesta
        response = ModelMetricsResponse(
            accuracy=metadata['accuracy'],
            f2_score=metadata['f2_score'],
            training_history=training_history_data,
            confusion_matrix=confusion_matrix_data,
            training_date=metadata.get('training_date'),
            training_samples=metadata.get('training_samples'),
            validation_samples=metadata.get('validation_samples')
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener métricas del modelo: {str(e)}"
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
