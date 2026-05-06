from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
import io
import soundfile as sf
import numpy as np
import tempfile

from services.audio_service import AudioService
from services.embedding_service import EmbeddingService
from services.tts_service import TTSService


app = FastAPI()

audio_service = AudioService()
tts_service = TTSService()
embedding_service = EmbeddingService(
    tts_service.tts.synthesizer.tts_model
)

@app.post("/tts")
async def tts(
    text: str = Form(...),
    speaker: str = Form(...),
    audio: UploadFile = File(None) 
):
    if audio is not None:
        audio_bytes = await audio.read()

        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)

        chunks = audio_service.preprocess("temp_audio.wav")
        embedding = embedding_service.get_or_create(speaker, chunks)

    else:
        embedding = embedding_service.get_or_create(speaker, None)

    out = tts_service.infer(text, embedding)
    wav = out["wav"] if isinstance(out, dict) else out

    wav = audio_service.postprocess(wav)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tts_service.tts.synthesizer.save_wav(wav, tmp.name)

        with open(tmp.name, "rb") as f:
            output_bytes = f.read()

    return Response(
        content=output_bytes,
        media_type="audio/wav"
    )