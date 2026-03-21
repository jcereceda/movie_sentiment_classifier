# 🎬 Movie Sentiment Analysis  
## Sistema de Clasificación de Sentimientos con Machine Learning

---

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema completo de análisis de sentimientos para reseñas de películas, combinando técnicas avanzadas de Machine Learning con una arquitectura moderna de microservicios.

El sistema utiliza modelos de lenguaje pre-entrenados (**BERT**) junto con redes neuronales personalizadas para clasificar automáticamente reseñas como positivas o negativas, proporcionando además métricas de confianza detalladas.

La solución está diseñada para ser **escalable, mantenible y fácilmente desplegable en entornos cloud**, con capacidades de reentrenamiento automático y gestión inteligente de datos.

---

## 🎯 Objetivos del Proyecto

### ✅ Objetivos Principales

- Clasificación Precisa de Sentimientos (>85% accuracy)
- Sistema de Reentrenamiento Automático
- API REST Escalable
- Gestión Inteligente de Datos

### ➕ Objetivos Secundarios

- Visualizaciones interactivas de métricas  
- Sistema de feedback de usuarios  
- Tiempo de respuesta < 2 segundos  
- Procesamiento en lote de reseñas  

---

## 🔬 Metodología

### 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────┐
│         Frontend (Angular 21)           │
│  - Interfaz de usuario interactiva      │
│  - Visualización de métricas            │
│  - Sistema de feedback                  │
└─────────────────┬───────────────────────┘
                  │ HTTP/REST
┌─────────────────▼───────────────────────┐
│         Backend (FastAPI)               │
│  - API REST                             │
│  - Gestión de predicciones              │
│  - Scheduler de reentrenamiento         │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
┌───────▼────────┐  ┌──────▼──────────┐
│   MongoDB      │  │  Modelos ML     │
│  - Reseñas     │  │  - BERT         │
│  - Feedback    │  │  - Clasificador │
└────────────────┘  └─────────────────┘
```

---

### 🤖 Pipeline de Machine Learning

#### 1. Preprocesamiento de Datos

```python
def clean_text(text):
    text = text.lower()
    text = text.replace('‚', "'").replace('"', '"')
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    contractions = {
        "won't": "will not",
        "can't": "cannot",
        "n't": " not"
    }

    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)

    return text
```

---

#### 2. Generación de Embeddings

```python
inputs = tokenizer(
    text,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

outputs = model(**inputs, output_hidden_states=True)
cls_embedding = outputs.hidden_states[-1][:, 0, :]
```

---

#### 3. Arquitectura del Clasificador

```
Input (768)
   ↓
Dense (256) + ReLU + Dropout(0.4)
   ↓
Dense (128) + ReLU + Dropout(0.4)
   ↓
Output (1) + Sigmoid
```

---

#### 4. Entrenamiento

- Épocas: 20  
- Batch size: 64  
- Early stopping: 4  
- Split: 80/20  

---

#### 5. Reentrenamiento Automático

- Programación flexible  
- Validación automática  
- Backup de modelos  
- Recarga sin downtime  

---

## 📊 Resultados

### Métricas

- Accuracy: 85.23%  
- F2-Score: 84.12%  
- Precision: 86.5%  
- Recall: 83.8%  

### Matriz de Confusión

|                | Pred. Neg | Pred. Pos |
|----------------|----------|----------|
| Real Negativo  | 450      | 50       |
| Real Positivo  | 48       | 452      |

---

### Rendimiento API

- Predicción individual: ~150ms  
- Batch: ~800ms  
- Throughput: ~100 req/s  

---

## 🚀 Endpoints API

- POST /predict  
- POST /predict/batch  
- GET /model/info  
- GET /model/metrics  

---

## 📦 Instalación

```bash
git clone https://github.com/tu-usuario/movie-sentiment-api.git
cd movie-sentiment-api

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env

python migrate_data.py
python main.py
python app.py
```

---

## 🔮 Trabajo Futuro

- Clasificación multiclase  
- Soporte multilingüe  
- Explicabilidad (LIME/SHAP)
- Mejora de la web

