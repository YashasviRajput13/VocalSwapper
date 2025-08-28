import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from django.conf import settings
from django.core.files.storage import default_storage
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from gtts import gTTS
from googletrans import Translator
from pydub import AudioSegment
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from monai.inferers import SlidingWindowInferer
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity
from config import Config
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import inch
from datetime import datetime

# Initialize core components
logger = logging.getLogger(__name__)
translator = Translator()
os.makedirs(Config.AUDIO_CACHE, exist_ok=True)


LANGUAGE_MAPPING = {
    'en-US': 'en',
    'hi-IN': 'hi',
    'es-ES': 'es',
    'fr-FR': 'fr',
    'zh-CN': 'zh-cn'
}


# Add this DummyImagingModel class definition
class DummyImagingModel:
    def __call__(self, image_path):
        return {
            'abnormalities': [],
            'confidence': 0.0,
            'recommendation': 'Consult radiologist for detailed analysis'
        }

# Ensure output directories exist
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.AUDIO_CACHE, exist_ok=True)
os.makedirs(Config.MEDIA_ROOT, exist_ok=True)

# Initialize core components
logger = logging.getLogger(__name__)
translator = Translator()

class MedicalProcessor:
    """Core medical analysis processing engine"""
    def __init__(self):
        try:
            self.symptom_analyzer = pipeline(
                "text-classification",
                model="bvanaken/clinical-assertion-negation-bert"
            )
            self.ner_model = pipeline(
                "ner",
                model="samrawal/bert-base-uncased_clinical-ner",
                aggregation_strategy="simple"
            )
            self.imaging_model = self._load_imaging_model()
        except Exception as e:
            logger.error(f"Failed to initialize medical processor: {str(e)}")
            # Fallback to dummy models
            self.symptom_analyzer = self.dummy_text_analyzer
            self.ner_model = self.dummy_ner_model
            self.imaging_model = DummyImagingModel()

    def dummy_text_analyzer(self, text):
        return [{'label': 'NEGATIVE', 'score': 0.99}]
    
    def dummy_ner_model(self, text):
        return [{'word': 'symptom', 'score': 0.9, 'entity': 'PROBLEM'}]
    
    def _load_imaging_model(self):
        """Load medical imaging model"""
        model_path = os.path.join(Config.MODEL_DIR, "imaging_model.pt")
        if os.path.exists(model_path):
            return torch.jit.load(model_path)
        logger.warning("Using dummy imaging model")
        return DummyImagingModel()

    def analyze_text(self, text):
        """Perform comprehensive medical text analysis"""
        try:
            entities = self.ner_model(text)
            symptoms = self.symptom_analyzer(text)
            return {
                'entities': entities,
                'symptoms': symptoms,
                'diagnosis': self._generate_diagnosis(entities, symptoms)
            }
        except Exception as e:
            logger.error(f"Medical analysis failed: {str(e)}")
            return {'error': 'Medical analysis failed'}

    def _generate_diagnosis(self, entities, symptoms):
        """Generate diagnostic summary"""
        # Simplified diagnosis logic
        conditions = [e['word'] for e in entities if e['entity_group'] == 'PROBLEM']
        if conditions:
            return f"Potential conditions detected: {', '.join(conditions)}. Please consult a specialist."
        return "No specific conditions detected. Continue monitoring symptoms."

    def generate_audio(self, text, lang):
        """Generate and cache audio responses"""
        lang = lang.split('-')[0]
        lang = lang if lang in ['en','hi','es','fr','zh-cn','de','ja','ru'] else 'en'
        
        hash_key = hashlib.md5(f"{lang}-{text}".encode()).hexdigest()
        audio_file = f"{hash_key}.mp3"
        audio_path = os.path.join(Config.AUDIO_CACHE, audio_file)
        
        if not os.path.exists(audio_path):
            try:
                tts = gTTS(text=text, lang=lang)
                tts.save(audio_path)
            except Exception as e:
                logger.error(f"Audio generation failed: {str(e)}")
                return os.path.join(Config.AUDIO_CACHE, "error.mp3")
        
        return audio_path

def preprocess_text(text):
    """Clean and normalize medical text input"""
    if not text or not isinstance(text, str):
        return ""
    
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def generate_pdf_report(analysis_result, image_path):
    """Generate PDF medical report"""
    try:
        report_path = os.path.join(Config.MEDIA_ROOT, f"report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf")
        doc = SimpleDocTemplate(report_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        story.append(Paragraph("Medical Analysis Report", styles['Title']))
        story.append(Spacer(1, 12))
        
        # Patient Information
        story.append(Paragraph("Patient Information", styles['Heading2']))
        story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph("Name: [Patient Name]", styles['Normal']))
        story.append(Paragraph("Age: [Age]", styles['Normal']))
        story.append(Paragraph("Gender: [Gender]", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Image
        if image_path and os.path.exists(image_path):
            img = Image(image_path, width=4*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 12))
        
        # Results
        story.append(Paragraph("Analysis Results", styles['Heading2']))
        for condition, confidence in analysis_result.items():
            story.append(Paragraph(f"{condition.replace('_', ' ').title()}: {confidence*100:.2f}%", styles['Normal']))
        
        # Diagnosis
        story.append(Spacer(1, 12))
        story.append(Paragraph("Diagnostic Summary", styles['Heading2']))
        diagnosis = "Based on the analysis, potential abnormalities were detected. Please consult with a specialist."
        story.append(Paragraph(diagnosis, styles['Normal']))
        
        # Generate PDF
        doc.build(story)
        return report_path
    except Exception as e:
        logger.error(f"PDF generation failed: {str(e)}")
        return None

@csrf_exempt
def analyze(request):
    if request.method != 'POST' or 'file' not in request.FILES:
        return JsonResponse({'error': 'Invalid request'}, status=400)

    upload = request.FILES['file']
    dest = os.path.join(Config.MEDIA_ROOT, upload.name)
    with open(dest, 'wb') as f:
        for chunk in upload.chunks():
            f.write(chunk)

    try:
        output = subprocess.check_output(
            [sys.executable, "inference.py",
             '--model', 'output/models/model_ep100.pth',
             f'--xray={dest}'],
            cwd=Config.BASE_DIR,
        )
        result = json.loads(output)
        
        # Generate PDF report
        report_path = generate_pdf_report(result, dest)
        report_url = f"{settings.MEDIA_URL}{os.path.basename(report_path)}" if report_path else None
        
        return JsonResponse({
            'analysis': result,
            'report_url': report_url
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def start_training(request):
    """Initiate model training process"""
    open(Config.LOG_DIR, 'w').close()
    def run():
        with open(Config.LOG_DIR, 'a') as log:
            proc = subprocess.Popen(
                [sys.executable, "train.py"],
                cwd=Config.BASE_DIR,
                stdout=log, stderr=log
            )
            proc.wait()
    threading.Thread(target=run, daemon=True).start()
    return JsonResponse({'status': 'training_started'})

def logs(request):
    """Stream training logs in real-time"""
    def stream():
        with open(Config.LOG_DIR) as f:
            f.seek(0, os.SEEK_END)
            while True:
                if line := f.readline():
                    yield line + '\n'
                time.sleep(0.5)
    return StreamingHttpResponse(stream(), content_type='text/plain')



@csrf_exempt
def voice_query(request):
    """Process voice-based medical queries with real-time translation"""
    try:
        data = json.loads(request.body)
        user_query = data.get('query', '')
        src_lang = data.get('source_lang', 'auto')
        tgt_lang = data.get('target_lang', 'en')
        report_data = data.get('report_data', None)

        # Normalize language codes
        if src_lang in LANGUAGE_MAPPING:
            src_lang = LANGUAGE_MAPPING[src_lang]
        if tgt_lang in LANGUAGE_MAPPING:
            tgt_lang = LANGUAGE_MAPPING[tgt_lang]

        # Preprocess and translate query
        cleaned_query = preprocess_text(user_query)
        translated_query = translator.translate(
            cleaned_query, 
            src=src_lang if src_lang != 'auto' else None,
            dest='en'
        ).text

        # Perform medical analysis
        medical_processor = MedicalProcessor()
        analysis = medical_processor.analyze_text(translated_query)
        
        # Generate response
        if report_data:
            diagnosis = "Based on your medical report, I recommend consulting a specialist."
        else:
            diagnosis = analysis.get('diagnosis', 'Analysis completed')
        
        # Translate response to target language
        try:
            translated_response = translator.translate(
                diagnosis, 
                src='en',
                dest=tgt_lang
            ).text
        except Exception as e:
            logger.error(f"Response translation failed: {str(e)}")
            translated_response = diagnosis  # Use English response as fallback

        # Generate audio response
        audio_file = medical_processor.generate_audio(translated_response, tgt_lang)
        audio_url = f'{settings.MEDIA_URL}audio_cache/{os.path.basename(audio_file)}'
        
        return JsonResponse({
            'status': 'success',
            'response': translated_response,
            'audio_url': audio_url,
            'analysis': analysis
        })
    except Exception as e:
        logger.error(f"Voice processing failed: {str(e)}", exc_info=True)
        return JsonResponse({
            'error': 'Processing failed',
            'details': str(e)
        }, status=500)
def index(request):
    return render(request, 'index.html')

def xray_mode(request):
    return render(request, 'models/xray.html')
def ct_scan_mode(request):
    return render(request, 'models/ct_scan.html')
def mri_mode(request):
    return render(request, 'models/mri.html')
def ultrasound_mode(request):
    return render(request, 'models/ultrasound.html')

def serve_audio(request, filename):
    audio_path = os.path.join(Config.AUDIO_CACHE, filename)
    if os.path.exists(audio_path):
        with open(audio_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type="audio/mpeg")
            response['Content-Disposition'] = f'inline; filename="{filename}"'
            return response
    return HttpResponse(status=404)



# Add to imports
# from models import ConversationTranslator

from .translator import ConversationTranslator

# Initialize conversation translator
conversation_translator = ConversationTranslator()

# Add new view for conversation mode
@csrf_exempt
def conversation_mode(request):
    return render(request, 'conversation.html')

# Add new API for conversation translation
@csrf_exempt
def translate_conversation(request):
    try:
        data = json.loads(request.body)
        text = data.get('text', '')
        speaker_lang = data.get('speaker_lang', 'en')
        listener_lang = data.get('listener_lang', 'en')
        
        if not text:
            return JsonResponse({'error': 'Empty text'}, status=400)
        
        # Translate text
        translated_text = conversation_translator.translate(
            text, speaker_lang, listener_lang
        )
        
        # Generate audio
        medical_processor = MedicalProcessor()
        audio_file = medical_processor.generate_audio(translated_text, listener_lang)
        audio_url = f'{settings.MEDIA_URL}audio_cache/{os.path.basename(audio_file)}'
        
        return JsonResponse({
            'status': 'success',
            'translated_text': translated_text,
            'audio_url': audio_url
        })
    except Exception as e:
        logger.error(f"Conversation translation failed: {str(e)}")
        return JsonResponse({'error': 'Translation failed'}, status=500)