"""
Módulo para reentrenamiento automático del modelo
"""
import torch
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
import shutil

import config
from data.preprocessing import load_and_preprocess_data
from data.dataset import create_data_loaders, create_embedding_loaders
from models.embeddings import EmbeddingGenerator
from models.classifier import MovieSentimentClassifier
from training.trainer import ModelTrainer
from evaluation.evaluator import ModelEvaluator


class AutoRetrainer:
    """Gestor de reentrenamiento automático del modelo"""
    
    def __init__(self):
        self.current_model_path = config.CLASSIFIER_MODEL_PATH
        self.current_metadata_path = config.MODEL_METADATA_PATH
        self.backup_dir = os.path.join(config.MODEL_DIR, "backups")
        
        # Crear directorio de backups
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def get_current_accuracy(self):
        """
        Obtiene la accuracy del modelo actual
        
        Returns:
            float: Accuracy actual o 0.0 si no existe modelo
        """
        if not os.path.exists(self.current_metadata_path):
            print("⚠️  No existe modelo previo, se entrenará uno nuevo")
            return 0.0
        
        try:
            with open(self.current_metadata_path, 'r') as f:
                metadata = json.load(f)
            
            accuracy = metadata.get('accuracy', 0.0)
            print(f"📊 Accuracy del modelo actual: {accuracy:.4f}")
            return accuracy
        
        except Exception as e:
            print(f"⚠️  Error al leer metadata: {e}")
            return 0.0
    
    def backup_current_model(self):
        """
        Crea un backup del modelo actual con timestamp
        
        Returns:
            str: Ruta del backup creado
        """
        if not os.path.exists(self.current_model_path):
            print("ℹ️  No hay modelo actual para respaldar")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"model_backup_{timestamp}"
        backup_path = os.path.join(self.backup_dir, backup_name)
        
        os.makedirs(backup_path, exist_ok=True)
        
        # Copiar modelo y metadata
        shutil.copy2(
            self.current_model_path,
            os.path.join(backup_path, "classifier_model.pth")
        )
        
        if os.path.exists(self.current_metadata_path):
            shutil.copy2(
                self.current_metadata_path,
                os.path.join(backup_path, "model_metadata.json")
            )
        
        print(f"💾 Backup creado: {backup_path}")
        return backup_path
    
    def train_new_model(self, seed=None):
        """
        Entrena un nuevo modelo desde cero
        
        Args:
            seed: Semilla para reproducibilidad
            
        Returns:
            dict: Resultados del entrenamiento (accuracy, f2_score, etc.)
        """
        if seed is None:
            seed = config.SEED
        
        print("\n" + "="*80)
        print("INICIANDO REENTRENAMIENTO DEL MODELO")
        print("="*80)
        
        # Configurar semilla
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # 1. Cargar datos desde MongoDB
        print("\n📥 Cargando datos desde MongoDB...")
        try:
            df = load_and_preprocess_data(use_mongodb=True, seed=seed)
        except ValueError as e:
            print(f"❌ Error: {e}")
            return None
        
        print(f"✅ Datos cargados: {len(df)} reseñas")
        
        # 2. Crear DataLoaders
        train_loader, val_loader, train_df, val_df = create_data_loaders(
            df=df,
            text_column='clean_review',
            batch_size=config.BATCH_SIZE,
            test_size=config.TRAIN_TEST_SPLIT,
            seed=seed
        )
        
        # 3. Generar embeddings
        print("\n🔄 Generando embeddings con BERT...")
        embedding_generator = EmbeddingGenerator(
            model_name=config.PRETRAINED_MODEL_NAME,
            device=config.DEVICE,
            max_length=config.MAX_LENGTH
        )
        
        train_embeddings, train_labels = embedding_generator.generate_embeddings(train_loader)
        val_embeddings, val_labels = embedding_generator.generate_embeddings(val_loader)
        
        # 4. Crear DataLoaders con embeddings
        train_loader_emb, val_loader_emb = create_embedding_loaders(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            val_embeddings=val_embeddings,
            val_labels=val_labels,
            batch_size=config.EMBEDDING_BATCH_SIZE
        )
        
        # 5. Crear modelo
        print("\n🏗️  Creando nuevo modelo...")
        model = MovieSentimentClassifier(
            input_dim=config.INPUT_DIM,
            hidden_dim_1=config.HIDDEN_DIM_1,
            hidden_dim_2=config.HIDDEN_DIM_2,
            dropout_rate=config.DROPOUT_RATE
        ).to(config.DEVICE)
        
        # 6. Entrenar
        print("\n🎯 Entrenando modelo...")
        trainer = ModelTrainer(
            model=model,
            device=config.DEVICE,
            learning_rate=config.LEARNING_RATE,
            patience=config.PATIENCE
        )
        
        trainer.setup_loss_function(train_labels)
        
        # Guardar temporalmente en una ruta diferente
        temp_model_path = os.path.join(config.MODEL_DIR, "temp_model.pth")
        
        history = trainer.train(
            train_loader=train_loader_emb,
            val_loader=val_loader_emb,
            epochs=config.EPOCHS,
            model_save_path=temp_model_path
        )
        
        # 7. Evaluar
        print("\n📈 Evaluando modelo...")
        model.load_state_dict(torch.load(temp_model_path))
        model.eval()
        
        evaluator = ModelEvaluator(model, config.DEVICE)
        results = evaluator.evaluate(val_loader_emb)
        
        # 8. Preparar metadata
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
            "data_source": "MongoDB",
            "training_samples": len(train_df),
            "validation_samples": len(val_df)
        }
        
        results['metadata'] = metadata
        results['temp_model_path'] = temp_model_path
        
        print(f"\n✅ Nuevo modelo entrenado - Accuracy: {results['accuracy']:.4f}")
        
        return results
    
    def should_update_model(self, new_accuracy, current_accuracy, min_improvement=0.001):
        """
        Determina si el nuevo modelo debe reemplazar al actual
        
        Args:
            new_accuracy: Accuracy del nuevo modelo
            current_accuracy: Accuracy del modelo actual
            min_improvement: Mejora mínima requerida
            
        Returns:
            bool: True si debe actualizarse
        """
        improvement = new_accuracy - current_accuracy
        
        print(f"\n📊 Comparación de modelos:")
        print(f"   Modelo actual:  {current_accuracy:.4f}")
        print(f"   Modelo nuevo:   {new_accuracy:.4f}")
        print(f"   Mejora:         {improvement:+.4f}")
        print(f"   Mínimo requerido: {min_improvement:.4f}")
        
        if improvement > min_improvement:
            print(f"✅ El nuevo modelo es mejor (mejora de {improvement:.4f})")
            return True
        else:
            print(f"❌ El nuevo modelo no supera el umbral de mejora")
            return False
    
    def update_model(self, new_results):
        """
        Actualiza el modelo en producción con el nuevo modelo
        
        Args:
            new_results: Resultados del nuevo modelo entrenado
        """
        temp_model_path = new_results['temp_model_path']
        metadata = new_results['metadata']
        
        print("\n🔄 Actualizando modelo en producción...")
        
        # 1. Crear backup del modelo actual
        self.backup_current_model()
        
        # 2. Reemplazar modelo
        shutil.move(temp_model_path, self.current_model_path)
        print(f"✅ Modelo actualizado: {self.current_model_path}")
        
        # 3. Guardar nueva metadata
        with open(self.current_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Metadata actualizada: {self.current_metadata_path}")
        
        # 4. Limpiar archivos temporales
        self._cleanup_temp_files()
    
    def _cleanup_temp_files(self):
        """Limpia archivos temporales del entrenamiento"""
        temp_files = [
            os.path.join(config.MODEL_DIR, "temp_model.pth"),
            os.path.join(config.MODEL_DIR, "temp_metadata.json")
        ]
        
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def cleanup_old_backups(self, keep_last_n=5):
        """
        Elimina backups antiguos, manteniendo solo los N más recientes
        
        Args:
            keep_last_n: Número de backups a mantener
        """
        backups = sorted(
            [d for d in os.listdir(self.backup_dir) if d.startswith("model_backup_")],
            reverse=True
        )
        
        if len(backups) > keep_last_n:
            print(f"\n🗑️  Limpiando backups antiguos (manteniendo últimos {keep_last_n})...")
            
            for old_backup in backups[keep_last_n:]:
                backup_path = os.path.join(self.backup_dir, old_backup)
                shutil.rmtree(backup_path)
                print(f"   Eliminado: {old_backup}")
    
    def run_auto_retrain(self, min_improvement=0.001, force=False):
        """
        Ejecuta el proceso completo de reentrenamiento automático
        
        Args:
            min_improvement: Mejora mínima de accuracy requerida
            force: Si True, actualiza el modelo aunque no mejore
            
        Returns:
            dict: Resultado del proceso
        """
        print("\n" + "="*80)
        print("🤖 PROCESO DE REENTRENAMIENTO AUTOMÁTICO")
        print("="*80)
        print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        result = {
            "success": False,
            "model_updated": False,
            "message": "",
            "current_accuracy": 0.0,
            "new_accuracy": 0.0
        }
        
        try:
            # 1. Obtener accuracy actual
            current_accuracy = self.get_current_accuracy()
            result["current_accuracy"] = current_accuracy
            
            # 2. Entrenar nuevo modelo
            new_results = self.train_new_model()
            
            if new_results is None:
                result["message"] = "Error al entrenar el nuevo modelo"
                return result
            
            new_accuracy = new_results['accuracy']
            result["new_accuracy"] = new_accuracy
            
            # 3. Decidir si actualizar
            should_update = force or self.should_update_model(
                new_accuracy, 
                current_accuracy, 
                min_improvement
            )
            
            if should_update:
                # 4. Actualizar modelo
                self.update_model(new_results)
                result["model_updated"] = True
                result["message"] = f"Modelo actualizado exitosamente (accuracy: {new_accuracy:.4f})"
                
                # 5. Limpiar backups antiguos
                self.cleanup_old_backups(keep_last_n=5)
            else:
                # Limpiar archivos temporales
                self._cleanup_temp_files()
                result["message"] = "Modelo no actualizado (no supera umbral de mejora)"
            
            result["success"] = True
            
        except Exception as e:
            result["message"] = f"Error durante el reentrenamiento: {str(e)}"
            print(f"\n❌ {result['message']}")
        
        print("\n" + "="*80)
        print("📋 RESUMEN DEL REENTRENAMIENTO")
        print("="*80)
        print(f"Estado: {'✅ Exitoso' if result['success'] else '❌ Fallido'}")
        print(f"Modelo actualizado: {'Sí' if result['model_updated'] else 'No'}")
        print(f"Accuracy anterior: {result['current_accuracy']:.4f}")
        print(f"Accuracy nuevo: {result['new_accuracy']:.4f}")
        print(f"Mensaje: {result['message']}")
        
        return result
