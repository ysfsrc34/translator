from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer
import os

app = FastAPI(
    title="Translator API",
    description="İngilizce'den Türkçe'ye çeviri yapan API",
    version="1.0.0"
)

# Model yükleme - singleton pattern
class TranslatorModel:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            model_name = "Helsinki-NLP/opus-mt-tc-big-en-tr"
            cls._instance = {
                'tokenizer': MarianTokenizer.from_pretrained(model_name),
                'model': MarianMTModel.from_pretrained(model_name)
            }
        return cls._instance

class TranslationRequest(BaseModel):
    text: str
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Hello, how are you?"
            }
        }

@app.post("/translate", 
    response_description="Çeviri sonucu",
    responses={
        400: {"description": "Geçersiz girdi"},
        500: {"description": "Sunucu hatası"}
    }
)
async def translate(request: TranslationRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Boş metin çevrilemez")
            
        translator = TranslatorModel.get_instance()
        input_text = request.text
        translated = translator['model'].generate(
            **translator['tokenizer']([input_text], return_tensors="pt", padding=True)
        )
        translated_text = translator['tokenizer'].decode(translated[0], skip_special_tokens=True)
        return {"translated_text": translated_text, "status": "success"}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Translator API is running"}