from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Ana sayfa URL’si
    path('video_search/', views.video_search, name='video_search'),
    path('video_comments/<str:video_id>/', views.video_comments, name='video_comments'),
    path('video_models/',views.video_models,name="video_models"),
    path('model_results/', views.model_results, name='model_results'), 
]