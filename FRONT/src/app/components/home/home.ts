import { Component, OnInit, signal, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatIconModule } from '@angular/material/icon';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { Movie } from '../../services/movie';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatCardModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatProgressSpinnerModule,
    MatIconModule,
    MatSnackBarModule
  ],
  templateUrl: './home.html',
  styleUrls: ['./home.scss']
})
export class HomeComponent implements OnInit {
  private movieService = inject(Movie);
  private snackBar = inject(MatSnackBar);

  // Signals
  movies = signal<any[]>([]);
  currentMovieIndex = signal(0);
  reviewText = signal('');
  sentimentResult = signal<any>(null);
  isLoading = signal(false);
  feedbackGiven = signal(false);

  // Computed signals
  currentMovie = computed(() => {
    const moviesList = this.movies();
    const index = this.currentMovieIndex();
    return moviesList[index] || null;
  });

  currentPoster = computed(() => {
    const movie = this.currentMovie();
    return movie ? `https://image.tmdb.org/t/p/w500${movie.poster_path}` : '';
  });

  currentTitle = computed(() => {
    const movie = this.currentMovie();
    return movie ? movie.title : '';
  });

  currentOverview = computed(() => {
    const movie = this.currentMovie();
    return movie ? movie.overview : '';
  });

  sentimentColor = computed(() => {
    const result = this.sentimentResult();
    if (!result) return '';
    return result.sentiment === 'Positive' ? 'green' : 'red';
  });

  ngOnInit(): void {
    this.loadMovies();
  }

  loadMovies(): void {
    this.movieService.getPopularMovies().subscribe({
      next: (response) => {
        if (response.results && Array.isArray(response.results)) {
          this.movies.set(response.results);
        }
      },
      error: (error) => {
        console.error('Error loading movies:', error);
        this.movies.set([
          {
            title: 'Sample Movie 1',
            poster_path: null,
            overview: 'This is a sample movie description.'
          },
          {
            title: 'Sample Movie 2',
            poster_path: null,
            overview: 'This is another sample movie description.'
          }
        ]);
      }
    });
  }

  nextMovie(): void {
    const moviesList = this.movies();
    if (moviesList.length > 0) {
      this.currentMovieIndex.update(index =>
        (index + 1) % moviesList.length
      );
      // Limpiar todo al cambiar de película
      this.resetReviewState();
    }
  }

  resetReviewState(): void {
    this.reviewText.set('');
    this.sentimentResult.set(null);
    this.feedbackGiven.set(false);
  }

  analyzeSentiment(): void {
    const review = this.reviewText();
    if (!review.trim()) return;

    this.isLoading.set(true);
    this.feedbackGiven.set(false);

    this.movieService.predictSentiment(review).subscribe({
      next: (response) => {
        this.sentimentResult.set(response);
        this.isLoading.set(false);
      },
      error: (error) => {
        console.error('Error analyzing sentiment:', error);
        this.isLoading.set(false);
      }
    });
  }

  giveFeedback(isExpected: boolean): void {
    if (this.feedbackGiven()) {
      this.snackBar.open('Feedback already submitted', 'Close', {
        duration: 2000
      });
      return;
    }

    const movie = this.currentMovie();
    const result = this.sentimentResult();
    const review = this.reviewText();

    if (!movie || !result || !review) return;

    // Preparar datos para insertar en BD
    const feedbackData = {
      review: review,
      sentiment: isExpected ? result.sentiment.toLowerCase() :
                (result.sentiment === 'Positive' ? 'negative' : 'positive')
    };

    this.movieService.insertReview(feedbackData).subscribe({
      next: (response) => {
        this.feedbackGiven.set(true);
        this.snackBar.open(
          isExpected ? 'Thank you! Feedback saved successfully' :
                      'Thank you! Your correction has been saved',
          'Close',
          { duration: 3000 }
        );
      },
      error: (error) => {
        console.error('Error saving feedback:', error);
        this.snackBar.open('Error saving feedback. Please try again.', 'Close', {
          duration: 3000
        });
      }
    });
  }
}
