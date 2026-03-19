"""
Script para migrar datos desde CSV a MongoDB
"""
import sys
from data.database import migrate_csv_to_mongodb
import config


def main():
    """Ejecuta la migración de datos"""
   
    print("="*80)
    print("MIGRACIÓN DE DATOS CSV → MONGODB")
    print("="*80)
   
    csv_path = config.DATA_PATH
   
    print(f"\nArchivo CSV: {csv_path}")
    print(f"MongoDB URL: {config.MONGODB_URL}")
    print(f"Base de datos: {config.MONGODB_DATABASE}")
    print(f"Colección: {config.MONGODB_COLLECTION}")
   
    try:
        migrate_csv_to_mongodb(csv_path, clean_data=True)
        print("\n✅ Migración completada exitosamente")
        print("\nAhora puedes entrenar el modelo con: python main.py")
       
    except FileNotFoundError:
        print(f"\n❌ Error: No se encontró el archivo {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error durante la migración: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()