"""
Módulo para generación de embeddings con BERT
"""
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification


class EmbeddingGenerator:
    """Generador de embeddings usando BERT pre-entrenado"""
    
    def __init__(self, model_name, device, max_length=128):
        """
        Args:
            model_name: Nombre del modelo pre-entrenado
            device: Dispositivo (cuda/cpu)
            max_length: Longitud máxima de secuencia
        """
        self.device = device
        self.max_length = max_length
        
        print(f"Cargando modelo BERT: {model_name}...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        print(f"Configuracion del modelo: {device}")
        print(f" - hidden size: {self.model.config.hidden_size}") #debe de ser 768
        print(f" - Num hidden layers: {self.model.config.num_hidden_layers}")

        print(f"Modelo cargado en dispositivo: {device}")
    
    def generate_embeddings(self, dataloader):
        """
        Genera embeddings para un DataLoader
        
        Args:
            dataloader: DataLoader con los textos
            
        Returns:
            embeddings, labels (arrays de numpy)
        """
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (batch_texts, batch_labels) in enumerate(dataloader):
                print(f"Procesando batch {batch_idx+1}/{len(dataloader)}...", end="\r")
                
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs, output_hidden_states=True)
                cls_embeddings = outputs.hidden_states[-1][:, 0, :]

                all_embeddings.append(cls_embeddings.cpu().numpy())
                all_labels.append(torch.tensor(batch_labels).numpy())
        
        print("\nEmbeddings generados exitosamente.")
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        return embeddings, labels
