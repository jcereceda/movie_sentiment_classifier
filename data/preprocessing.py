"""
Módulo para preprocesamiento de texto
"""
import re
import string
import pandas as pd
from data.database import MongoDBHandler


def clean_text(text):
    """
    Limpia y normaliza el texto de las reseñas
   
    Args:
        text: Texto a limpiar
       
    Returns:
        Texto limpio y normalizado
    """
    html_escape = re.compile('<.*?>')
   
    if not isinstance(text, str):
        text = str(text)
   
    text = text.lower()
    text = text.replace('‚', "'").replace('“', '"').replace('”', '"').replace('’', "'")
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(html_escape, '', text)
    text = re.sub(r'\d+', '', text)
   
    punct_to_remove = string.punctuation.replace('?', '').replace('!', '')
    text = text.translate(str.maketrans('', '', punct_to_remove))
    text = re.sub(r'\s+', ' ', text).strip()
   
    return text


def load_and_preprocess_data(filepath=None, seed=1, use_mongodb=True):
    """
    Carga y preprocesa el dataset de reseñas
   
    Args:
        filepath: Ruta al archivo CSV (si use_mongodb=False)
        seed: Semilla para reproducibilidad
        use_mongodb: Si cargar desde MongoDB o CSV
       
    Returns:
        DataFrame preprocesado
    """
    if use_mongodb:
        print("Cargando datos desde MongoDB...")
        db_handler = MongoDBHandler()
        df = db_handler.get_all_reviews()
        db_handler.close()
       
        if df.empty:
            raise ValueError(
                "No se encontraron datos en MongoDB. "
                "Ejecuta la migración primero con: python migrate_data.py"
            )
       
        # Verificar si ya tiene clean_review
        if 'clean_review' not in df.columns or df['clean_review'].isna().any():
            print("Aplicando preprocesamiento de texto...")
            df['clean_review'] = df['review'].apply(clean_text)
    else:
        print(f"Cargando datos desde {filepath}...")
       
        df = pd.read_csv(
            filepath,
            sep=None,
            engine='python',
            encoding='utf-8'
        )
       
        df.columns = ["review", "sentiment"]
       
        df["sentiment_val"] = df["sentiment"].map({
            "negative": 0,
            "positive": 1
        })
       
        # Limpieza de datos
        df["review"] = df["review"].astype(str).str.replace('\ufeff', '', regex=False)
        df["sentiment"] = df["sentiment"].astype(str).replace(["nan", "NaN", "None", ""], " ")
        df["sentiment_val"] = pd.to_numeric(df["sentiment_val"], errors='coerce')
       
        # Filtrar filas válidas
        df = df[df['sentiment_val'].isin([0, 1])].copy()
       
        print(f"Aplicando preprocesamiento de texto...")
        df['clean_review'] = df['review'].apply(clean_text)
   
    print(f"Datos cargados: {len(df)} reseñas")
    print(f"Distribución de sentimientos:\n{df['sentiment_val'].value_counts()}")
   
    return df


def show_data_samples(df, n_samples=3, seed=1):
    """
    Muestra ejemplos de reseñas por categoría
   
    Args:
        df: DataFrame con las reseñas
        n_samples: Número de muestras a mostrar
        seed: Semilla para reproducibilidad
    """
    print("\n" + "="*80)
    print("EJEMPLOS DE RESEÑAS")
    print("="*80)
   
    print("\nRESEÑAS NEGATIVAS:")
    print("-"*50)
    negative = df[df["sentiment_val"] == 0]
    if len(negative) > 0:
        for review in negative["review"].sample(min(n_samples, len(negative)), random_state=seed):
            print(f"• {review[:150]}...")
   
    print("\nRESEÑAS POSITIVAS:")
    print("-"*50)
    positive = df[df["sentiment_val"] == 1]
    if len(positive) > 0:
        for review in positive["review"].sample(min(n_samples, len(positive)), random_state=seed):
            print(f"• {review[:150]}...")