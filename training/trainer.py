"""
Módulo para entrenamiento del modelo
"""
import torch
import torch.nn as nn
import torch.optim as optim


class ModelTrainer:
    """Entrenador del modelo con Early Stopping"""
    
    def __init__(self, model, device, learning_rate=0.001, patience=4):
        """
        Args:
            model: Modelo a entrenar
            device: Dispositivo (cuda/cpu)
            learning_rate: Tasa de aprendizaje
            patience: Épocas de paciencia para early stopping
        """
        self.model = model
        self.device = device
        self.patience = patience
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = None
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.early_stop_counter = 0
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def setup_loss_function(self, train_labels):
        """
        Configura la función de pérdida con pesos de clase
        
        Args:
            train_labels: Etiquetas de entrenamiento para calcular pesos
        """
        num_negative = (train_labels == 0).sum()
        num_positive = (train_labels == 1).sum()
        
        pos_weight = None
        if num_negative > 0 and num_positive > 0:
            pos_weight_value = num_negative / num_positive
            pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(self.device)
            print(f"Peso para clase positiva: {pos_weight_value:.2f}")
        
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def train_epoch(self, train_loader):
        """
        Entrena una época
        
        Args:
            train_loader: DataLoader de entrenamiento
            
        Returns:
            Pérdida promedio de la época
        """
        self.model.train()
        train_loss = 0.0
        
        for batch_emb, batch_lbl in train_loader:
            batch_emb, batch_lbl = batch_emb.to(self.device), batch_lbl.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(batch_emb)
            loss = self.criterion(logits, batch_lbl)
            
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        
        return train_loss / len(train_loader)
    
    def validate(self, val_loader):
        """
        Valida el modelo
        
        Args:
            val_loader: DataLoader de validación
            
        Returns:
            val_loss, val_accuracy
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_emb, batch_lbl in val_loader:
                batch_emb, batch_lbl = batch_emb.to(self.device), batch_lbl.to(self.device)
                
                logits = self.model(batch_emb)
                loss = self.criterion(logits, batch_lbl)
                val_loss += loss.item()
                
                preds_prob = torch.sigmoid(logits)
                preds = (preds_prob > 0.5).float()
                correct += (preds == batch_lbl).sum().item()
                total += batch_lbl.size(0)
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def train(self, train_loader, val_loader, epochs, model_save_path):
        """
        Entrena el modelo completo
        
        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            epochs: Número de épocas
            model_save_path: Ruta para guardar el mejor modelo
            
        Returns:
            Historial de entrenamiento
        """
        print("\n" + "="*80)
        print("INICIANDO ENTRENAMIENTO")
        print("="*80)
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")
            
            # Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), model_save_path)
                print("  → Mejor modelo guardado")
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.patience:
                    print(f"Early stopping en epoch {epoch+1}")
                    break
        
        print(f"\nEntrenamiento finalizado. Mejor epoch: {self.best_epoch} "
              f"(Val Loss: {self.best_val_loss:.4f})")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss
        }
