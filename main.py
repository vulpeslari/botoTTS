from boto_engine import BotoTTS
from audio_processing import audio_process, split_audio, convert

boto = BotoTTS()
path = "marina_sena.mp3"
speaker = path[:-4]

audio, sr = convert(path)

chunks = split_audio(audio, sr)

processed_chunks = []

for chunk in chunks:
    processed = audio_process(chunk, sr)

    if processed is not None:
        processed_chunks.append(processed)

boto.generate(
    audio=processed_chunks,
    text="geladeira cachorro feliz triste cama música",
    speaker=speaker,
)
