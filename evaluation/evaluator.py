"""
Módulo para evaluación del modelo
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, fbeta_score, confusion_matrix


class ModelEvaluator:
    """Evaluador del modelo entrenado"""
    
    def __init__(self, model, device):
        """
        Args:
            model: Modelo a evaluar
            device: Dispositivo (cuda/cpu)
        """
        self.model = model
        self.device = device
    
    def evaluate(self, val_loader):
        """
        Evalúa el modelo en el conjunto de validación
        
        Args:
            val_loader: DataLoader de validación
            
        Returns:
            Diccionario con métricas
        """
        self.model.eval()
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for emb, lbl in val_loader:
                emb = emb.to(self.device)
                logits = self.model(emb)
                preds_prob = torch.sigmoid(logits)
                preds = (preds_prob > 0.5).float().cpu().numpy().flatten()
                
                all_preds.extend(preds)
                all_true.extend(lbl.numpy().flatten())
        
        acc = accuracy_score(all_true, all_preds)
        f2 = fbeta_score(all_true, all_preds, beta=2)
        cm = confusion_matrix(all_true, all_preds)
        tn, fp, fn, tp = cm.ravel()
        
        results = {
            'accuracy': acc,
            'f2_score': f2,
            'confusion_matrix': cm,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'predictions': all_preds,
            'true_labels': all_true
        }
        
        return results
    
    def print_results(self, results):
        """
        Imprime los resultados de evaluación
        
        Args:
            results: Diccionario con métricas
        """
        print("\n" + "="*60)
        print("EVALUACIÓN FINAL EN VALIDACIÓN")
        print("="*60)
        print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"F2-Score: {results['f2_score']:.4f}")
        print(f"\nMatriz de confusión:")
        print(f"  TN (Negativo bien detectado): {results['true_negatives']}")
        print(f"  FP (Negativo predicho como Positivo): {results['false_positives']}")
        print(f"  FN (Positivo predicho como Negativo): {results['false_negatives']}")
        print(f"  TP (Positivo bien detectado): {results['true_positives']}")


class SentimentPredictor:
    """Predictor para nuevas reseñas"""
    
    def __init__(self, bert_model, classifier_model, tokenizer, device, max_length=128):
        """
        Args:
            bert_model: Modelo BERT para embeddings
            classifier_model: Clasificador entrenado
            tokenizer: Tokenizador de BERT
            device: Dispositivo (cuda/cpu)
            max_length: Longitud máxima de secuencia
        """
        self.bert_model = bert_model
        self.classifier_model = classifier_model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        
        self.bert_model.eval()
        self.classifier_model.eval()
    
    def predict(self, review_text):
        """
        Predice el sentimiento de una reseña
        
        Args:
            review_text: Texto de la reseña
            
        Returns:
            sentiment, probability
        """
        inputs = self.tokenizer(
            review_text,
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
            prob = torch.sigmoid(logits)
            prediction = (prob > 0.5).long()
        
        sentiment = "Positive" if prediction.item() == 1 else "Negative"
        probability = prob.item()
        
        return sentiment, probability
