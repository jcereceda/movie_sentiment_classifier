"""
Módulo para visualización de resultados
"""
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, title='Matriz de Confusión'):
    """
    Visualiza la matriz de confusión
    
    Args:
        cm: Matriz de confusión
        title: Título del gráfico
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive']
    )
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    """
    Visualiza el historial de entrenamiento
    
    Args:
        history: Diccionario con historial de entrenamiento
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Pérdidas
    ax1.plot(history['train_losses'], label='Train Loss', marker='o')
    ax1.plot(history['val_losses'], label='Validation Loss', marker='s')
    ax1.axvline(x=history['best_epoch']-1, color='r', linestyle='--', label='Best Epoch')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    ax1.set_title('Evolución de la Pérdida')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['val_accuracies'], label='Validation Accuracy', marker='o', color='green')
    ax2.axvline(x=history['best_epoch']-1, color='r', linestyle='--', label='Best Epoch')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Evolución del Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
