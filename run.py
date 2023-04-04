import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from faster_whisper import WhisperModel

app = FastAPI()
model_size = "large-v2"
# model_size = "tiny"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Save the uploaded file to disk
    if file.content_type.split('/')[0] != 'audio':
        raise HTTPException(
            status_code=400, detail="Invalid file type, please upload an audio file")

    # Save the uploaded file to disk
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Transcribe the file using faster-whisper
    segments, info = model.transcribe(file.filename, beam_size=5)
    print("Detected language '%s' with probability %f" %
          (info.language, info.language_probability))

    # for segment in segments:
    #     print("[%.2fs -> %.2fs] %s" %
    #           (segment.start, segment.end, segment.text))

    segments_list = [{"start": segment.start, "end": segment.end,
                      "text": segment.text} for segment in segments]
    
    print(segments_list)

    return {"segments": segments_list, "language": info.language, "language_probability": info.language_probability}