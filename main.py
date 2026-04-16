from services.audio_service import AudioService
from services.embedding_service import EmbeddingService
from services.tts_service import TTSService
import os

def generate_speech(audio_input, text, speaker):
    audio_service = AudioService()
    tts_service = TTSService()
    
    embedding_service = EmbeddingService(
        tts_service.tts.synthesizer.tts_model
    )

    # 1. Pré-processamento de áudio
    chunks = audio_service.preprocess(audio_input)

    # 2. Recupera ou cria embedding do speaker
    embedding = embedding_service.get_or_create(speaker, chunks)

    # 3. Inferência do modelo TTS
    out = tts_service.infer(text, embedding)
    wav = out["wav"] if isinstance(out, dict) else out

    # 4. Pós-processamento 
    wav = audio_service.postprocess(wav)

    os.makedirs("voices", exist_ok=True)
    output_path = f"voices/{speaker}.wav"
    tts_service.tts.synthesizer.save_wav(wav, output_path)

    return output_path


if __name__ == "__main__":
    path = "larissa.aac"
    speaker = "larissa"

    output = generate_speech(
        audio_input=path,
        text="geladeira cachorro feliz triste cama música",
        speaker=speaker,
    )