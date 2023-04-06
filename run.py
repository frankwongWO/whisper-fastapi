import os
import shutil
import datetime
from werkzeug.utils import secure_filename
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from faster_whisper import WhisperModel
from translate import Translator
from pydantic import BaseModel


app = FastAPI()

# Set the directory for uploading files
today = datetime.date.today()
date_str = today.strftime('%Y-%m-%d')
UPLOAD_FOLDER = f'./file/{date_str}'

@app.post("/transcribe")
async def transcribe(model_size: str = Form(...), device: str = Form(...), compute_type: str = Form(...),to_lang: str =  Form(None), file: UploadFile = File(...)):
    # Check if the file is an audio file
    ALLOWED_FILE_TYPES = {'audio', 'video'}
    if file.content_type.split('/')[0] not in ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400, detail="Invalid file type, please upload an audio or video file")

    # Check if the directory exists and create it if not
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # The path to the build object file
    filename = secure_filename(file.filename)
    target_path = os.path.join(UPLOAD_FOLDER, filename)

    # Save the file to the target path
    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # # Save the uploaded file to disk
    # with open(file.filename, "wb") as buffer:
    #     shutil.copyfileobj(file.file, buffer)

    # model_size = tiny, small, medium, large-v2,
    model_size = model_size

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Run on GPU with FP16
    # model = WhisperModel(model_size, device="cuda", compute_type="float16")
    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    # Transcribe the file using faster-whisper
    segments, info = model.transcribe(target_path, beam_size=5)
    print("Detected language '%s' with probability %f" %
          (info.language, info.language_probability))
    
    print(to_lang)

    # Translate the segments text
    
    if(to_lang != None):
        translator = Translator(from_lang=info.language,to_lang=to_lang)
        segments_list = [{"start": segment.start, "end": segment.end,
                      "text": translator.translate(segment.text)} for segment in segments]
    else:
        segments_list = [{"start": segment.start, "end": segment.end,
                      "text": segment.text} for segment in segments]
        
        
    text = ','.join([segment['text'] for segment in segments_list])
    print(text)

    print(segments_list)

    return {"text": text,"segments": segments_list, "language": info.language, "language_probability": info.language_probability }

class TranslationRequest(BaseModel):
    text: str
    to_lang: str
    from_lang :str

@app.post("/translate")
async def translate_text(req: TranslationRequest):
    # Translate the text to the specified language
    translator = Translator(from_lang=req.from_lang,to_lang=req.to_lang)
    translation = translator.translate(req.text)

    return {"text": translation}