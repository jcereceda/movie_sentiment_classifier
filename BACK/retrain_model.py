"""
Script para ejecutar reentrenamiento manual del modelo
"""
import argparse
from training.auto_retrain import AutoRetrainer


def main():
    parser = argparse.ArgumentParser(description='Reentrenar modelo de clasificación')
    parser.add_argument(
        '--min-improvement',
        type=float,
        default=0.001,
        help='Mejora mínima de accuracy requerida (default: 0.001)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Forzar actualización aunque no mejore'
    )
    
    args = parser.parse_args()
    
    retrainer = AutoRetrainer()
    result = retrainer.run_auto_retrain(
        min_improvement=args.min_improvement,
        force=args.force
    )
    
    exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()
