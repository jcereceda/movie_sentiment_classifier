import { Component, OnInit, signal, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { BaseChartDirective } from 'ng2-charts';
import { ChartConfiguration } from 'chart.js';
import { Movie } from '../../services/movie';

@Component({
  selector: 'app-stats',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatProgressSpinnerModule,
    BaseChartDirective
  ],
  templateUrl: './stats.html',
  styleUrls: ['./stats.scss']
})
export class StatsComponent implements OnInit {
  private movieService = inject(Movie);

  // Signals
  modelMetrics = signal<any>(null);
  isLoading = signal(true);

  // Chart configurations
  lineChartData = signal<ChartConfiguration['data']>({
    datasets: [],
    labels: []
  });

  lineChartOptions: ChartConfiguration['options'] = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top'
      },
      title: {
        display: true,
        text: 'Training History'
      }
    }
  };

  barChartData = signal<ChartConfiguration['data']>({
    datasets: [],
    labels: []
  });

  barChartOptions: ChartConfiguration['options'] = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top'
      },
      title: {
        display: true,
        text: 'Confusion Matrix'
      }
    }
  };

  ngOnInit(): void {
    this.loadModelStats();
  }

  loadModelStats(): void {
    this.movieService.getModelStats().subscribe({
      next: (response) => {
        this.modelMetrics.set(response);
        this.prepareCharts();
        this.isLoading.set(false);
      },
      error: (error) => {
        console.error('Error loading model stats:', error);
        this.isLoading.set(false);
      }
    });
  }

  prepareCharts(): void {
    const metrics = this.modelMetrics();
    if (!metrics) return;

    // Training History Chart
    const history = metrics.training_history;
    if (history) {
      this.lineChartData.set({
        labels: history.epochs.map((e: number) => `Epoch ${e}`),
        datasets: [
          {
            data: history.train_losses,
            label: 'Train Loss',
            borderColor: 'rgb(255, 99, 132)',
            backgroundColor: 'rgba(255, 99, 132, 0.1)',
            tension: 0.4
          },
          {
            data: history.val_losses,
            label: 'Validation Loss',
            borderColor: 'rgb(54, 162, 235)',
            backgroundColor: 'rgba(54, 162, 235, 0.1)',
            tension: 0.4
          },
          {
            data: history.val_accuracies,
            label: 'Validation Accuracy',
            borderColor: 'rgb(75, 192, 192)',
            backgroundColor: 'rgba(75, 192, 192, 0.1)',
            tension: 0.4
          }
        ]
      });
    }

    // Confusion Matrix Chart
    const cm = metrics.confusion_matrix;
    if (cm) {
      this.barChartData.set({
        labels: ['True Negatives', 'False Positives', 'False Negatives', 'True Positives'],
        datasets: [
          {
            data: [
              cm.true_negatives,
              cm.false_positives,
              cm.false_negatives,
              cm.true_positives
            ],
            label: 'Count',
            backgroundColor: [
              'rgba(75, 192, 192, 0.6)',
              'rgba(255, 99, 132, 0.6)',
              'rgba(255, 159, 64, 0.6)',
              'rgba(54, 162, 235, 0.6)'
            ],
            borderColor: [
              'rgb(75, 192, 192)',
              'rgb(255, 99, 132)',
              'rgb(255, 159, 64)',
              'rgb(54, 162, 235)'
            ],
            borderWidth: 2
          }
        ]
      });
    }
  }
}
