from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
import soundfile as sf

from services.audio_service import AudioService
from services.embedding_service import EmbeddingService
from services.tts_service import TTSService

app = FastAPI()

audio_service = AudioService()
tts_service = TTSService()
embedding_service = EmbeddingService(
    tts_service.tts.synthesizer.tts_model
)

class Request(BaseModel):
    text: str
    speaker: str


def generate_audio_stream(text, speaker):
    # 🔥 gera embedding (cache já resolve repetição)
    chunks = audio_service.preprocess("larissa.aac")
    embedding = embedding_service.get_or_create(speaker, chunks)

    # 🔥 inferência
    out = tts_service.infer(text, embedding)
    wav = out["wav"] if isinstance(out, dict) else out

    wav = audio_service.postprocess(wav)

    # 🔥 transforma em stream
    buffer = io.BytesIO()
    sf.write(buffer, wav, 22050, format="WAV")
    buffer.seek(0)

    yield buffer.read()


@app.get("/tts")
def tts(text: str, speaker: str):
    return StreamingResponse(
        generate_audio_stream(text, speaker),
        media_type="audio/wav"
    )