# core/models.py
from django.db import models

# Create your models here.
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class ConversationTranslator:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.load_models()
    
    def load_models(self):
        # Load models for different language pairs
        pairs = [
            ('hi', 'en'), ('en', 'hi'),
            ('es', 'en'), ('en', 'es'),
            ('fr', 'en'), ('en', 'fr'),
            ('hi', 'es'), ('es', 'hi'),
           
        ]
        
        for src, tgt in pairs:
            key = f"{src}-{tgt}"
            model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
            try:
                self.models[key] = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.tokenizers[key] = AutoTokenizer.from_pretrained(model_name)
                print(f"Loaded model for {src}->{tgt}")
            except:
                # Use English as pivot for unsupported pairs
                self.models[key] = None
                print(f"Using pivot translation for {src}->{tgt}")
    
    def translate(self, text, src_lang, tgt_lang):
        if src_lang == tgt_lang:
            return text
        
        direct_key = f"{src_lang}-{tgt_lang}"
        
        # Try direct translation
        if direct_key in self.models and self.models[direct_key]:
            tokenizer = self.tokenizers[direct_key]
            model = self.models[direct_key]
            
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            outputs = model.generate(**inputs)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Use English as pivot for unsupported pairs
        if f"{src_lang}-en" in self.models and f"en-{tgt_lang}" in self.models:
            # Translate to English first
            intermediate = self.translate(text, src_lang, "en")
            # Then to target language
            return self.translate(intermediate, "en", tgt_lang)
        
        return text  
    
