"""
Módulo para manejo de datasets de PyTorch
"""
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


class MovieReviewDataset(Dataset):
    """Dataset personalizado para reseñas de películas"""
    
    def __init__(self, dataframe, text_column, label_column='sentiment_val'):
        """
        Args:
            dataframe: DataFrame con los datos
            text_column: Nombre de la columna con el texto
            label_column: Nombre de la columna con las etiquetas
        """
        self.texts = dataframe[text_column].values
        self.labels = dataframe[label_column].values
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def create_data_loaders(df, text_column, batch_size, test_size=0.2, seed=1):
    """
    Crea DataLoaders para entrenamiento y validación
    
    Args:
        df: DataFrame con los datos
        text_column: Columna con el texto preprocesado
        batch_size: Tamaño del batch
        test_size: Proporción para validación
        seed: Semilla para reproducibilidad
        
    Returns:
        train_loader, val_loader, train_df, val_df
    """
    print(f"\nDividiendo datos (train/val: {int((1-test_size)*100)}/{int(test_size*100)})...")
    
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df['sentiment_val']
    )
    
    print(f"Train: {len(train_df)} reseñas ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation: {len(val_df)} reseñas ({len(val_df)/len(df)*100:.1f}%)")
    
    train_dataset = MovieReviewDataset(train_df, text_column)
    val_dataset = MovieReviewDataset(val_df, text_column)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    
    return train_loader, val_loader, train_df, val_df


def create_embedding_loaders(train_embeddings, train_labels, val_embeddings, val_labels, batch_size):
    """
    Crea DataLoaders a partir de embeddings pre-calculados
    
    Args:
        train_embeddings: Embeddings de entrenamiento
        train_labels: Etiquetas de entrenamiento
        val_embeddings: Embeddings de validación
        val_labels: Etiquetas de validación
        batch_size: Tamaño del batch
        
    Returns:
        train_loader, val_loader
    """
    train_embeddings_tensor = torch.tensor(train_embeddings, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
    
    val_embeddings_tensor = torch.tensor(val_embeddings, dtype=torch.float32)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32).unsqueeze(1)
    
    train_dataset = TensorDataset(train_embeddings_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_embeddings_tensor, val_labels_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
