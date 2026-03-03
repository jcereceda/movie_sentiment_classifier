"""
Script principal para entrenar el clasificador de sentimientos
"""
import numpy as np
import torch
import json
from datetime import datetime

import config
from data.preprocessing import load_and_preprocess_data, show_data_samples
from data.dataset import create_data_loaders, create_embedding_loaders
from models.embeddings import EmbeddingGenerator
from models.classifier import MovieSentimentClassifier
from training.trainer import ModelTrainer
from evaluation.evaluator import ModelEvaluator
from utils.visualization import plot_confusion_matrix, plot_training_history


def set_seed(seed):
    """Configura la semilla para reproducibilidad"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_model_metadata(results, history):
    """Guarda metadata del modelo entrenado"""
    metadata = {
        "training_date": datetime.now().isoformat(),
        "accuracy": float(results['accuracy']),
        "f2_score": float(results['f2_score']),
        "best_epoch": int(history['best_epoch']),
        "best_val_loss": float(history['best_val_loss']),
        "confusion_matrix": {
            "true_negatives": int(results['true_negatives']),
            "false_positives": int(results['false_positives']),
            "false_negatives": int(results['false_negatives']),
            "true_positives": int(results['true_positives'])
        },
        "model_config": {
            "pretrained_model": config.PRETRAINED_MODEL_NAME,
            "input_dim": config.INPUT_DIM,
            "hidden_dim_1": config.HIDDEN_DIM_1,
            "hidden_dim_2": config.HIDDEN_DIM_2,
            "dropout_rate": config.DROPOUT_RATE,
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.EMBEDDING_BATCH_SIZE
        },
        "data_source": "MongoDB"
    }
   
    with open(config.MODEL_METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
   
    print(f"\nMetadata guardada en: {config.MODEL_METADATA_PATH}")


def main():
    """Función principal del pipeline de entrenamiento"""
   
    print("="*80)
    print("ENTRENAMIENTO DEL CLASIFICADOR DE SENTIMIENTOS")
    print("="*80)
   
    # 1. Configuración inicial
    set_seed(config.SEED)
    print(f"\nDispositivo: {config.DEVICE}")
   
    # 2. Cargar y preprocesar datos desde MongoDB
    try:
        df = load_and_preprocess_data(use_mongodb=True, seed=config.SEED)
    except ValueError as e:
        print(f"\n❌ {e}")
        print("\nPara migrar los datos, ejecuta: python migrate_data.py")
        return
   
    show_data_samples(df, n_samples=3, seed=config.SEED)
   
    # 3. Crear DataLoaders
    train_loader, val_loader, train_df, val_df = create_data_loaders(
        df=df,
        text_column='clean_review',
        batch_size=config.BATCH_SIZE,
        test_size=config.TRAIN_TEST_SPLIT,
        seed=config.SEED
    )
   
    # 4. Generar embeddings con BERT
    print("\n" + "="*80)
    print("GENERACIÓN DE EMBEDDINGS")
    print("="*80)
   
    embedding_generator = EmbeddingGenerator(
        model_name=config.PRETRAINED_MODEL_NAME,
        device=config.DEVICE,
        max_length=config.MAX_LENGTH
    )
   
    print("\nGenerando embeddings para TRAIN...")
    train_embeddings, train_labels = embedding_generator.generate_embeddings(train_loader)
   
    print("\nGenerando embeddings para VALIDATION...")
    val_embeddings, val_labels = embedding_generator.generate_embeddings(val_loader)
   
    print(f"\nTrain embeddings shape: {train_embeddings.shape}")
    print(f"Validation embeddings shape: {val_embeddings.shape}")
   
    # 5. Crear DataLoaders con embeddings
    train_loader_emb, val_loader_emb = create_embedding_loaders(
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        val_embeddings=val_embeddings,
        val_labels=val_labels,
        batch_size=config.EMBEDDING_BATCH_SIZE
    )
   
    # 6. Crear y entrenar el modelo
    print("\n" + "="*80)
    print("CREACIÓN DEL MODELO")
    print("="*80)
   
    model = MovieSentimentClassifier(
        input_dim=config.INPUT_DIM,
        hidden_dim_1=config.HIDDEN_DIM_1,
        hidden_dim_2=config.HIDDEN_DIM_2,
        dropout_rate=config.DROPOUT_RATE
    ).to(config.DEVICE)
   
    print(f"Modelo creado y movido a: {config.DEVICE}")
   
    # 7. Entrenar
    trainer = ModelTrainer(
        model=model,
        device=config.DEVICE,
        learning_rate=config.LEARNING_RATE,
        patience=config.PATIENCE
    )
   
    trainer.setup_loss_function(train_labels)
   
    history = trainer.train(
        train_loader=train_loader_emb,
        val_loader=val_loader_emb,
        epochs=config.EPOCHS,
        model_save_path=config.CLASSIFIER_MODEL_PATH
    )
   
    # 8. Cargar el mejor modelo
    model.load_state_dict(torch.load(config.CLASSIFIER_MODEL_PATH))
    model.eval()
   
    # 9. Evaluar
    evaluator = ModelEvaluator(model, config.DEVICE)
    results = evaluator.evaluate(val_loader_emb)
    evaluator.print_results(results)
   
    # 10. Guardar metadata
    save_model_metadata(results, history)
   
    # 11. Visualizaciones
    print("\nGenerando visualizaciones...")
    plot_confusion_matrix(results['confusion_matrix'])
    plot_training_history(history)
   
    print("\n" + "="*80)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*80)
    print(f"\nModelo guardado en: {config.CLASSIFIER_MODEL_PATH}")
    print(f"Metadata guardada en: {config.MODEL_METADATA_PATH}")
    print("\nPara iniciar la API, ejecuta: python app.py")


if __name__ == "__main__":
    main()