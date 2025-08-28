from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('analyze/', views.analyze, name='analyze'),
    path('start-training/', views.start_training, name='start_training'),
    path('logs/', views.logs, name='logs'),

    path('voice-query/', views.voice_query, name='voice_query'),
     path('audio/<str:filename>/', views.serve_audio, name='serve_audio'),

    # path('audio/<str:filename>/', views.serve_audio, name='serve_audio'),
    path('conversation/', views.conversation_mode, name='conversation_mode'),
    path('translate-conversation/', views.translate_conversation, name='translate_conversation'),


    path('x-ray/', views.xray_mode, name='x-ray'),

    path('ct-scan/', views.ct_scan_mode, name='ct-scan'),
    path('mri/', views.mri_mode, name='mri'),
    path('ultrasound/', views.ultrasound_mode, name='ultrasound'),


]


