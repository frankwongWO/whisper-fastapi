import os
import io
import shutil
import datetime
from werkzeug.utils import secure_filename
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from translate import Translator
from pydantic import BaseModel
import base64
import time


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set the directory for uploading files
today = datetime.date.today()
date_str = today.strftime("%Y-%m-%d")
UPLOAD_FOLDER = f"./file/{date_str}"


@app.post("/transcribe")
async def transcribe(
    model_size: str = Form(...),
    compute_type: str = Form(...),
    device: str = Form("cuda"),
    to_lang: str = Form(None),
    file: UploadFile = File(...),
    beam_size: int = Form(5),
):
    start_time = time.time()  # 记录开始时间
    print(
        f"content_type: {file.content_type}\n"
        f"to_lang: {to_lang}\n"
        f"model_size: {model_size}\n"
        f"compute_type: {compute_type}\n"
        f"device: {device}\n"
        f"beam_size: {beam_size}\n"
    )
    # Check if the file is an audio file
    ALLOWED_FILE_TYPES = {
        "audio/mpeg",
        "video/mp4",
        "audio/mp3",
        "audio/ogg",
        "application/octet-stream",
    }
    if file.content_type not in ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type, please upload an audio or video file",
        )

    # file.content_type.split("/")[0]

    # Check if the directory exists and create it if not
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # The path to the build object file
    filename = secure_filename(file.filename)
    target_path = os.path.join(UPLOAD_FOLDER, filename)

    # Save the file to the target path
    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # model_size = tiny, small,base, medium, large-v2,
    # compute_type= "int8", "int8_float16", "float16", "None"

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Transcribe the file using faster-whisper
    segments, info = model.transcribe(target_path, beam_size=beam_size)

    print(
        "Detected language '%s' with probability %f"
        % (info.language, info.language_probability)
    )

    translator = Translator(from_lang=info.language, to_lang=to_lang)
    # Translate the segments text
    if to_lang != None:
        segments_list = [
            {
                "start": segment.start,
                "end": segment.end,
                "text": translator.translate(segment.text),
            }
            for segment in segments
        ]
    else:
        segments_list = [
            {"start": segment.start, "end": segment.end, "text": segment.text}
            for segment in segments
        ]

    text = ",".join([segment["text"] for segment in segments_list])

    print("text", text)
    print("segments_list", segments_list)

    end_time = time.time()  # 记录结束时间
    processing_time = end_time - start_time  # 计算处理时间

    print("processing_time", processing_time)

    return {
        "text": text,
        "segments": segments_list,
        "language": info.language,
        "language_probability": info.language_probability,
        "processing_time": processing_time,  # 返回处理时间
    }


class base64DataTranscribe_Request(BaseModel):
    file: str
    model_size: str
    compute_type: str
    device: str = "cuda"
    to_lang: str = None
    beam_size: int = 5


@app.post("/base64DataTranscribe")
async def base64DataTranscribe(req: base64DataTranscribe_Request):
    start_time = time.time()  # 记录开始时间
    print(
        f"base64Str: {req.file[:50]}....\n",
        f"to_lang: {req.to_lang}\n"
        f"model_size: {req.model_size}\n"
        f"compute_type: {req.compute_type}\n"
        f"device: {req.device}",
    )

    data = base64.b64decode(req.file)
    binary_data = io.BytesIO(data)

    model = WhisperModel(
        req.model_size, device=req.device, compute_type=req.compute_type
    )

    segments, info = model.transcribe(
        binary_data,
        beam_size=req.beam_size,
    )

    print(
        "Detected language '%s' with probability %f"
        % (info.language, info.language_probability)
    )

    translator = Translator(from_lang=info.language, to_lang=req.to_lang)
    if req.to_lang != None:
        segments_list = [
            {
                "start": segment.start,
                "end": segment.end,
                "text": translator.translate(segment.text),
            }
            for segment in segments
        ]
    else:
        segments_list = [
            {"start": segment.start, "end": segment.end, "text": segment.text}
            for segment in segments
        ]

    text = ",".join([segment["text"] for segment in segments_list])
    print("text", text)
    print("segments_list", segments_list)

    end_time = time.time()  # 记录结束时间
    processing_time = end_time - start_time  # 计算处理时间

    print("processing_time", processing_time)

    return {
        "text": text,
        "segments": segments_list,
        "language": info.language,
        "language_probability": info.language_probability,
        "processing_time": processing_time,  # 返回处理时间
    }


class Base64FastTranscribe_Request(BaseModel):
    file: str


@app.post("/base64FastTranscribe")
async def base64FastTranscribe(req: Base64FastTranscribe_Request):
    start_time = time.time()  # 记录开始时间
    data = base64.b64decode(req.file)
    binary_data = io.BytesIO(data)

    model = WhisperModel("small", device="cuda", compute_type="int8")

    segments, info = model.transcribe(
        binary_data, beam_size=1, without_timestamps=True, temperature=0, language="zh"
    )

    text = ""
    for item in segments:
        text += item[2]

    end_time = time.time()  # 记录结束时间
    processing_time = end_time - start_time  # 计算处理时间

    return {
        "text": text,
        "processing_time": processing_time,  # 返回处理时间
    }


class TranslationRequest(BaseModel):
    text: str
    to_lang: str
    from_lang: str


@app.post("/translate")
async def translate_text(req: TranslationRequest):
    # Translate the text to the specified language
    translator = Translator(from_lang=req.from_lang, to_lang=req.to_lang)
    translation = translator.translate(req.text)

    return {"text": translation}
