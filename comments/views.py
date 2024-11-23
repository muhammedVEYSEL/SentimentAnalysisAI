from django.shortcuts import render
from .models import VideoComment
import requests
import csv
import os
import re
import nltk
import json
import ast
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from django.views.decorators.csrf import csrf_protect
from django.http import JsonResponse



# API Anahtarınızı buraya ekleyin
API_KEY = ''  # Kendi API anahtarınızı girin

nltk.download('stopwords')
stop_words = set(stopwords.words('turkish'))

# Word2Vec Modeli
word2vec_model = Word2Vec(vector_size=100, min_count=1, window=5)  # Vektör boyutunu 100 olarak ayarladık

def preprocess_text(text):
    """
    Verilen metni temizleyip ön işleme yapar.
    """
    # Küçük harfe çevirme
    text = text.lower()
    # Noktalama işaretlerini kaldırma
    text = re.sub(r'[^\w\s]', '', text)
    # Stop words'leri kaldırma
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def get_word2vec_vector(text):
    """
    Metni Word2Vec kullanarak vektöre çevirir.
    """
    words = text.split()
    vector = np.mean([word2vec_model.wv[word] for word in words if word in word2vec_model.wv], axis=0)
    if isinstance(vector, np.ndarray):
        return vector
    else:
        return np.zeros(word2vec_model.vector_size)


def index(request):
    return render(request, 'index.html')

def video_search(request):
    video_name = request.GET.get('video_name')
    videos = []  # Çoklu video sonuçlarını saklayacağımız liste

    if video_name:
        # YouTube Search API ile video araması yap
        search_url = f'https://www.googleapis.com/youtube/v3/search'
        search_params = {
            'part': 'snippet',
            'q': video_name,
            'type': 'video',
            'key': API_KEY,
            'maxResults': 10
        }
        search_response = requests.get(search_url, params=search_params).json()

        if 'items' in search_response:
            for item in search_response['items']:
                video_id = item['id']['videoId']
                title = item['snippet']['title']
                description = item['snippet']['description']
                thumbnail = item['snippet']['thumbnails']['default']['url']
                published_at = item['snippet']['publishedAt'][:10]
                
                videos.append({
                    'video_id': video_id,
                    'title': title,
                    'description': description,
                    'thumbnail': thumbnail,
                    'published_at': published_at
                })

    context = {
        'videos': videos,
        'video_name': video_name
    }
    return render(request, 'video_results.html', context)

def video_comments(request, video_id):
    comments_url = f'https://www.googleapis.com/youtube/v3/commentThreads'
    comments_params = {
        'part': 'snippet',
        'videoId': video_id,
        'key': API_KEY,
        'maxResults': 20
    }
    comments_response = requests.get(comments_url, params=comments_params).json()

    comments = []

    if 'items' in comments_response:
        for item in comments_response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            author = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
            like_count = item['snippet']['topLevelComment']['snippet']['likeCount']
            # Metni temizle
            cleaned_comment = preprocess_text(comment)
            
            # Sayısallaştırma işlemi (Word2Vec vektörü oluşturma)
            vector = get_word2vec_vector(cleaned_comment)
            comments.append({
                'author': author,
                'comment': comment,
                'like_count': like_count,
                'vector': vector
            })

    # Yorumları CSV dosyasına kaydetme
    csv_file_path = os.path.join('new_data', f'{video_id}_comments.csv')
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['author', 'comment', 'like_count'] + [f'vector_{i}' for i in range(word2vec_model.vector_size)]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for comment in comments:
            row = {
                'author': comment['author'],
                'comment': comment['comment'],
                'like_count': comment['like_count'],
            }
            row.update({f'vector_{i}': val for i, val in enumerate(comment['vector'])})
            writer.writerow(row)     

    context = {
        'comments': comments,
        'video_id': video_id,
        'csv_file_path': csv_file_path,
    }
    return render(request, 'comment.html', context)

@csrf_protect
def video_models(request):
    if request.method == "POST":
        try:
            # Raw JSON verisini al
            data = json.loads(request.body.decode('utf-8'))
            
            comments = data.get("comments", [])
            video_title = data.get("video_title", "Bilinmeyen Video")
            video_id = data.get("video_id", "Bilinmiyor")

            if not comments:
                return render(request, 'error.html', {'message': 'Yorumlar boş geldi.'})

            # Modeli yükle ve yorumları analiz et
            tokenizer = BertTokenizer.from_pretrained('saved_bert_model4')
            model = TFBertForSequenceClassification.from_pretrained('saved_bert_model4')
            
            predictions = []
            for comment in comments:
                text = comment.get('comment', '')
                author = comment.get('author', 'Bilinmeyen Yazar')

                if text.strip() == "":
                    continue  # Boş yorumları atla

                encoded = tokenizer(text, truncation=True, padding=True, return_tensors='tf')
                outputs = model(**encoded)
                predicted_class = tf.argmax(outputs.logits, axis=-1).numpy()[0]

                predictions.append({
                    'author': author,
                    'text': text,
                    'class': int(predicted_class)
                })

            # Analiz sonuçlarını sonuç sayfasına gönder
            return JsonResponse({
                'predictions': predictions,
                'video_title': video_title,
                'video_id': video_id
            })

        except json.JSONDecodeError as e:
            print(f"JSON Hatası: {e}")
            return JsonResponse({'error': 'Geçersiz JSON verisi alındı.'}, status=400)

    return JsonResponse({'error': 'Yalnızca POST isteklerine izin veriliyor.'}, status=405)

def model_results(request):
    predictions = request.GET.get('predictions')
    if predictions:
       predictions = json.loads(predictions)
    video_title = request.GET.get('video_title', 'Bilinmeyen Video')
    video_id = request.GET.get('video_id', 'Bilinmiyor')

    return render(request, 'model_results.html', {
        'predictions': predictions,
        'video_title': video_title,
        'video_id': video_id,
    })