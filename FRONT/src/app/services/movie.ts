import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environments';

@Injectable({
  providedIn: 'root',
})
export class Movie {
  private apiUrl = environment.apiUrl;
  private tmdbKey = environment.tmdbApiKey;
  private tmdbBaseUrl = 'https://api.themoviedb.org/3';

  constructor(private http: HttpClient){
  }

  getPopularMovies(): Observable<any> {
    const headers = new HttpHeaders({
      'accept':'application/json',
      'Authorization': 'Bearer ' + this.tmdbKey
    });
    const numeroAleatorio: number = Math.floor(Math.random() * 10) + 1;
    return this.http.get(
      `${this.tmdbBaseUrl}/movie/popular?api_key=${this.tmdbKey}&language=en-US&page=${numeroAleatorio}`,
      { headers }
    );
  }

  predictSentiment(review: string):Observable<any> {
    return this.http.post(`${this.apiUrl}/predict`, {review});
  }

  getModelStats(): Observable<any> {
    return this.http.get(`${this.apiUrl}/model/metrics`);
  }

  insertReview(data: {review: string, sentiment: string}): Observable<any> {
    return this.http.post(`${this.apiUrl}/database/review`, data);
  }
 }
