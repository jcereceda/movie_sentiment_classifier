"""
Módulo para programación de tareas de reentrenamiento
"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime
import logging

from training.auto_retrain import AutoRetrainer


class RetrainingScheduler:
    """Programador de tareas de reentrenamiento automático"""
    
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.retrainer = AutoRetrainer()
        self.is_running = False
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def retrain_job(self, min_improvement=0.001):
        """
        Tarea de reentrenamiento que se ejecutará periódicamente
        
        Args:
            min_improvement: Mejora mínima requerida
        """
        self.logger.info("="*80)
        self.logger.info("Iniciando tarea programada de reentrenamiento")
        self.logger.info("="*80)
        
        try:
            result = self.retrainer.run_auto_retrain(min_improvement=min_improvement)
            
            if result['success']:
                if result['model_updated']:
                    self.logger.info(
                        f"✅ Modelo actualizado - "
                        f"Accuracy: {result['current_accuracy']:.4f} → {result['new_accuracy']:.4f}"
                    )
                else:
                    self.logger.info(
                        f"ℹ️  Modelo no actualizado - "
                        f"Accuracy actual ({result['current_accuracy']:.4f}) "
                        f"sigue siendo mejor"
                    )
            else:
                self.logger.error(f"❌ Error en reentrenamiento: {result['message']}")
        
        except Exception as e:
            self.logger.error(f"❌ Error crítico en tarea de reentrenamiento: {e}")
    
    def schedule_daily(self, hour=2, minute=0, min_improvement=0.001):
        """
        Programa reentrenamiento diario
        
        Args:
            hour: Hora del día (0-23)
            minute: Minuto de la hora (0-59)
            min_improvement: Mejora mínima requerida
        """
        trigger = CronTrigger(hour=hour, minute=minute)
        
        self.scheduler.add_job(
            func=self.retrain_job,
            trigger=trigger,
            args=[min_improvement],
            id='daily_retrain',
            name='Reentrenamiento Diario',
            replace_existing=True
        )
        
        self.logger.info(f"📅 Reentrenamiento programado diariamente a las {hour:02d}:{minute:02d}")
    
    def schedule_weekly(self, day_of_week='mon', hour=2, minute=0, min_improvement=0.001):
        """
        Programa reentrenamiento semanal
        
        Args:
            day_of_week: Día de la semana (mon, tue, wed, thu, fri, sat, sun)
            hour: Hora del día
            minute: Minuto
            min_improvement: Mejora mínima requerida
        """
        trigger = CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute)
        
        self.scheduler.add_job(
            func=self.retrain_job,
            trigger=trigger,
            args=[min_improvement],
            id='weekly_retrain',
            name='Reentrenamiento Semanal',
            replace_existing=True
        )
        
        self.logger.info(
            f"📅 Reentrenamiento programado semanalmente los {day_of_week} "
            f"a las {hour:02d}:{minute:02d}"
        )
    
    def schedule_interval(self, hours=24, min_improvement=0.001):
        """
        Programa reentrenamiento por intervalo
        
        Args:
            hours: Intervalo en horas
            min_improvement: Mejora mínima requerida
        """
        trigger = IntervalTrigger(hours=hours)
        
        self.scheduler.add_job(
            func=self.retrain_job,
            trigger=trigger,
            args=[min_improvement],
            id='interval_retrain',
            name=f'Reentrenamiento cada {hours}h',
            replace_existing=True
        )
        
        self.logger.info(f"⏰ Reentrenamiento programado cada {hours} horas")
    
    def start(self):
        """Inicia el scheduler"""
        if not self.is_running:
            self.scheduler.start()
            self.is_running = True
            self.logger.info("🚀 Scheduler iniciado")
            
            # Mostrar trabajos programados
            jobs = self.scheduler.get_jobs()
            if jobs:
                self.logger.info("Trabajos programados:")
                for job in jobs:
                    self.logger.info(f"  - {job.name}: {job.next_run_time}")
            else:
                self.logger.warning("⚠️  No hay trabajos programados")
    
    def stop(self):
        """Detiene el scheduler"""
        if self.is_running:
            self.scheduler.shutdown()
            self.is_running = False
            self.logger.info("🛑 Scheduler detenido")
    
    def trigger_manual_retrain(self, min_improvement=0.001):
        """
        Ejecuta un reentrenamiento manual inmediato
        
        Args:
            min_improvement: Mejora mínima requerida
        """
        self.logger.info("🔧 Reentrenamiento manual iniciado")
        self.retrain_job(min_improvement=min_improvement)
    
    def get_next_run_time(self):
        """
        Obtiene la próxima ejecución programada
        
        Returns:
            datetime o None
        """
        jobs = self.scheduler.get_jobs()
        if jobs:
            return min(job.next_run_time for job in jobs)
        return None
