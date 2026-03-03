"""
Módulo para manejo de MongoDB
"""
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, DuplicateKeyError
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import config


class MongoDBHandler:
    """Manejador de conexión y operaciones con MongoDB"""
   
    def __init__(self):
        """Inicializa la conexión a MongoDB"""
        self.client = None
        self.db = None
        self.collection = None
        self._connect()
   
    def _connect(self):
        """Establece la conexión con MongoDB"""
        try:
            print(f"Conectando a MongoDB: {config.MONGODB_URL}")
            self.client = MongoClient(
                config.MONGODB_URL,
                serverSelectionTimeoutMS=5000
            )
           
            # Verificar conexión
            self.client.admin.command('ping')
           
            self.db = self.client[config.MONGODB_DATABASE]
            self.collection = self.db[config.MONGODB_COLLECTION]
           
            # Crear índices
            self._create_indexes()
           
            print(f"✅ Conectado a MongoDB: {config.MONGODB_DATABASE}.{config.MONGODB_COLLECTION}")
           
        except ConnectionFailure as e:
            print(f"❌ Error al conectar con MongoDB: {e}")
            raise
   
    def _create_indexes(self):
        """Crea índices para optimizar consultas"""
        try:
            # Índice en sentiment_val para filtrado rápido
            self.collection.create_index([("sentiment_val", ASCENDING)])
           
            # Índice en created_at para ordenamiento temporal
            self.collection.create_index([("created_at", DESCENDING)])
           
            # Índice de texto para búsquedas en review
            self.collection.create_index([("review", "text")])
           
            print("✅ Índices creados correctamente")
        except Exception as e:
            print(f"⚠️  Advertencia al crear índices: {e}")
   
    def insert_review(self, review: str, sentiment: str, sentiment_val: int,
                     clean_review: Optional[str] = None) -> str:
        """
        Inserta una reseña en la base de datos
       
        Args:
            review: Texto original de la reseña
            sentiment: Sentimiento en texto ('positive' o 'negative')
            sentiment_val: Valor numérico del sentimiento (0 o 1)
            clean_review: Texto preprocesado (opcional)
           
        Returns:
            ID del documento insertado
        """
        document = {
            "review": review,
            "sentiment": sentiment,
            "sentiment_val": sentiment_val,
            "clean_review": clean_review or review,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
       
        result = self.collection.insert_one(document)
        return str(result.inserted_id)
   
    def insert_many_reviews(self, reviews: List[Dict]) -> List[str]:
        """
        Inserta múltiples reseñas en lote
       
        Args:
            reviews: Lista de diccionarios con los datos de las reseñas
           
        Returns:
            Lista de IDs insertados
        """
        for review in reviews:
            review["created_at"] = datetime.utcnow()
            review["updated_at"] = datetime.utcnow()
       
        result = self.collection.insert_many(reviews)
        return [str(id) for id in result.inserted_ids]
   
    def get_all_reviews(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Obtiene todas las reseñas como DataFrame
       
        Args:
            limit: Número máximo de documentos a retornar
           
        Returns:
            DataFrame con las reseñas
        """
        query = {}
        cursor = self.collection.find(query)
       
        if limit:
            cursor = cursor.limit(limit)
       
        reviews = list(cursor)
       
        if not reviews:
            print("⚠️  No se encontraron reseñas en la base de datos")
            return pd.DataFrame()
       
        df = pd.DataFrame(reviews)
       
        # Eliminar el campo _id de MongoDB si existe
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
       
        print(f"✅ Cargadas {len(df)} reseñas desde MongoDB")
        return df
   
    def get_reviews_by_sentiment(self, sentiment_val: int,
                                 limit: Optional[int] = None) -> pd.DataFrame:
        """
        Obtiene reseñas filtradas por sentimiento
       
        Args:
            sentiment_val: 0 para negativo, 1 para positivo
            limit: Número máximo de documentos
           
        Returns:
            DataFrame con las reseñas filtradas
        """
        query = {"sentiment_val": sentiment_val}
        cursor = self.collection.find(query)
       
        if limit:
            cursor = cursor.limit(limit)
       
        reviews = list(cursor)
        df = pd.DataFrame(reviews)
       
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
       
        return df
   
    def count_reviews(self) -> Dict[str, int]:
        """
        Cuenta el número de reseñas por sentimiento
       
        Returns:
            Diccionario con conteos
        """
        total = self.collection.count_documents({})
        positive = self.collection.count_documents({"sentiment_val": 1})
        negative = self.collection.count_documents({"sentiment_val": 0})
       
        return {
            "total": total,
            "positive": positive,
            "negative": negative
        }
   
    def search_reviews(self, text: str, limit: int = 10) -> pd.DataFrame:
        """
        Busca reseñas por texto
       
        Args:
            text: Texto a buscar
            limit: Número máximo de resultados
           
        Returns:
            DataFrame con resultados
        """
        query = {"$text": {"$search": text}}
        cursor = self.collection.find(query).limit(limit)
       
        reviews = list(cursor)
        df = pd.DataFrame(reviews)
       
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
       
        return df
   
    def delete_all_reviews(self) -> int:
        """
        Elimina todas las reseñas (usar con precaución)
       
        Returns:
            Número de documentos eliminados
        """
        result = self.collection.delete_many({})
        print(f"🗑️  Eliminadas {result.deleted_count} reseñas")
        return result.deleted_count
   
    def close(self):
        """Cierra la conexión con MongoDB"""
        if self.client:
            self.client.close()
            print("✅ Conexión con MongoDB cerrada")


def migrate_csv_to_mongodb(csv_path: str, clean_data: bool = True):
    """
    Migra datos desde un archivo CSV a MongoDB
   
    Args:
        csv_path: Ruta al archivo CSV
        clean_data: Si aplicar preprocesamiento antes de insertar
    """
    from data.preprocessing import clean_text
   
    print("\n" + "="*80)
    print("MIGRACIÓN DE CSV A MONGODB")
    print("="*80)
   
    # Cargar CSV
    print(f"\nCargando datos desde {csv_path}...")
    df = pd.read_csv(csv_path, sep=None, engine='python', encoding='utf-8')
    df.columns = ["review", "sentiment"]
   
    # Mapear sentimientos
    df["sentiment_val"] = df["sentiment"].map({
        "negative": 0,
        "positive": 1
    })
   
    # Limpiar datos
    df["review"] = df["review"].astype(str).str.replace('\ufeff', '', regex=False)
    df["sentiment"] = df["sentiment"].astype(str).replace(["nan", "NaN", "None", ""], " ")
    df["sentiment_val"] = pd.to_numeric(df["sentiment_val"], errors='coerce')
   
    # Filtrar filas válidas
    df = df[df['sentiment_val'].isin([0, 1])].copy()
   
    print(f"✅ Datos cargados: {len(df)} reseñas válidas")
   
    # Aplicar preprocesamiento si se solicita
    if clean_data:
        print("\nAplicando preprocesamiento...")
        df['clean_review'] = df['review'].apply(clean_text)
    else:
        df['clean_review'] = df['review']
   
    # Conectar a MongoDB
    db_handler = MongoDBHandler()
   
    # Verificar si ya existen datos
    counts = db_handler.count_reviews()
    if counts['total'] > 0:
        print(f"\n⚠️  La base de datos ya contiene {counts['total']} reseñas")
        response = input("¿Deseas eliminar los datos existentes? (s/n): ")
        if response.lower() == 's':
            db_handler.delete_all_reviews()
        else:
            print("Migración cancelada")
            db_handler.close()
            return
   
    # Preparar documentos para inserción
    print("\nInsertando datos en MongoDB...")
    documents = df.to_dict('records')
   
    # Insertar en lotes para mejor rendimiento
    batch_size = 1000
    total_inserted = 0
   
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        db_handler.insert_many_reviews(batch)
        total_inserted += len(batch)
        print(f"Progreso: {total_inserted}/{len(documents)} reseñas insertadas", end="\r")
   
    print(f"\n✅ Migración completada: {total_inserted} reseñas insertadas")
   
    # Mostrar estadísticas
    counts = db_handler.count_reviews()
    print("\nEstadísticas de la base de datos:")
    print(f"  Total: {counts['total']}")
    print(f"  Positivas: {counts['positive']} ({counts['positive']/counts['total']*100:.1f}%)")
    print(f"  Negativas: {counts['negative']} ({counts['negative']/counts['total']*100:.1f}%)")
   
    db_handler.close()