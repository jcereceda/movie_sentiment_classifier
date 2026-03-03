"""
Script de ejemplo para probar la API
"""
import requests
import json


API_URL = "http://localhost:8000"


def test_health():
    """Prueba el endpoint de salud"""
    print("\n" + "="*60)
    print("TEST: Health Check")
    print("="*60)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_single_prediction():
    """Prueba predicción individual"""
    print("\n" + "="*60)
    print("TEST: Single Prediction")
    print("="*60)
    
    reviews = [
        "This movie was absolutely fantastic! Best film of the year.",
        "Terrible movie, complete waste of time and money.",
        "It was okay, nothing special but not terrible either."
    ]
    
    for review in reviews:
        response = requests.post(
            f"{API_URL}/predict",
            json={"review": review}
        )
        
        print(f"\nReview: {review}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_batch_prediction():
    """Prueba predicción en lote"""
    print("\n" + "="*60)
    print("TEST: Batch Prediction")
    print("="*60)
    
    reviews = [
        "Amazing cinematography and brilliant acting!",
        "Boring plot and terrible dialogue.",
        "The special effects were incredible.",
        "I fell asleep halfway through.",
        "A masterpiece of modern cinema!"
    ]
    
    response = requests.post(
        f"{API_URL}/predict/batch",
        json={"reviews": reviews}
    )
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    
    print(f"\nTotal processed: {result['total_processed']}")
    print("\nResults:")
    for i, res in enumerate(result['results'], 1):
        print(f"\n{i}. Review: {res['review'][:50]}...")
        print(f"   Sentiment: {res['sentiment']}")
        print(f"   Confidence: {res['confidence']:.4f}")


def test_model_info():
    """Prueba endpoint de información del modelo"""
    print("\n" + "="*60)
    print("TEST: Model Info")
    print("="*60)
    
    response = requests.get(f"{API_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


if __name__ == "__main__":
    print("="*60)
    print("TESTING MOVIE SENTIMENT API")
    print("="*60)
    
    try:
        test_health()
        test_single_prediction()
        test_batch_prediction()
        test_model_info()
        
        print("\n" + "="*60)
        print("✅ TODOS LOS TESTS COMPLETADOS")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: No se pudo conectar a la API")
        print("Asegúrate de que la API esté corriendo en http://localhost:8000")
    except Exception as e:
        print(f"\n❌ Error: {e}")
