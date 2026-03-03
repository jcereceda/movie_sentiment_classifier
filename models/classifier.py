"""
Módulo con la arquitectura del clasificador
"""
import torch
import torch.nn as nn


class MovieSentimentClassifier(nn.Module):
    """Red neuronal para clasificación de sentimientos"""
    
    def __init__(self, input_dim=768, hidden_dim_1=128, hidden_dim_2=64, dropout_rate=0.3):
        """
        Args:
            input_dim: Dimensión de entrada (embedding)
            hidden_dim_1: Neuronas en primera capa oculta
            hidden_dim_2: Neuronas en segunda capa oculta
            dropout_rate: Tasa de dropout
        """
        super(MovieSentimentClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim_2, 1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Logits de salida
        """
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)
