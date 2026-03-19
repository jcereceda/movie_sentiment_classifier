#!/bin/bash

echo "==================================="
echo "Movie Sentiment Analysis - Startup"
echo "==================================="

# Verificar si existe el modelo entrenado
if [ ! -f "saved_models/classifier_model.pth" ]; then
    echo ""
    echo "⚠️  No se encontró modelo entrenado"
    echo "Iniciando entrenamiento..."
    echo ""
    python main.py
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Error en el entrenamiento"
        exit 1
    fi
fi

echo ""
echo "✅ Modelo encontrado"
echo "🚀 Iniciando API..."
echo ""

python api.py
