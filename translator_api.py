from fastapi import FastAPI
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer
import os

app = FastAPI(title="Translator API")

# Model yükleme
model_name = "Helsinki-NLP/opus-mt-tc-big-en-tr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

class TranslationRequest(BaseModel):
    text: str

@app.post("/translate")
async def translate(request: TranslationRequest):
    try:
        input_text = request.text
        translated = model.generate(
            **tokenizer([input_text], return_tensors="pt", padding=True)
        )
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        return {"translated_text": translated_text, "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "error"}

@app.get("/")
async def root():
    return {"message": "Translator API is running"}