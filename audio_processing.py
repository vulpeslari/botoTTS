import librosa
import noisereduce as nr
import numpy as np
import scipy.signal as signal
import pyloudnorm as pyln

TARGET_SR = 22050
TOP_DB = 30


def highpass(audio, sr, cutoff=100):
    b, a = signal.butter(5, cutoff / (sr / 2), btype="highpass")
    return signal.lfilter(b, a, audio)


def bandpass(audio, sr, low=120, high=7600):
    b, a = signal.butter(4, [low / (sr / 2), high / (sr / 2)], btype="band")
    return signal.lfilter(b, a, audio)


def convert(audio_input):
    if isinstance(audio_input, str):
        audio, sr = librosa.load(audio_input, sr=TARGET_SR, mono=True)
    else:
        audio, sr = audio_input

        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)

        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

    audio = audio.astype(np.float32)
  
    return audio, sr


def audio_process(
    audio,
    sr,
    apply_trim=True,
    apply_denoise=True
):
    audio, sr = convert((audio, sr))

    audio = highpass(audio, sr)

    if apply_denoise:
        audio = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=False,
            prop_decrease=0.7
        )

    audio = bandpass(audio, sr)

    if apply_trim:
        audio, _ = librosa.effects.trim(audio, top_db=TOP_DB)

    audio = librosa.util.normalize(audio)

    return audio

def audio_process_output(audio, sr):
    audio, sr = convert((audio, sr))  
    
    audio = highpass(audio, sr, cutoff=80)
    
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    
    if not np.isfinite(loudness):
        return audio
    
    audio = pyln.normalize.loudness(audio, loudness, -16.0)

    return audio

def split_audio(
    audio,
    sr,
    chunk_duration=5,
    overlap=1,
    min_speech_duration=1
):
    # detectar regiões com fala
    intervals = librosa.effects.split(audio, top_db=TOP_DB)

    speech_segments = []

    for start, end in intervals:
        segment = audio[start:end]

        duration = len(segment) / sr

        if duration < min_speech_duration:
            continue

        speech_segments.append(segment)

    chunk_size = int(chunk_duration * sr)
    step = int((chunk_duration - overlap) * sr)

    chunks = []

    for segment in speech_segments:
        for start in range(0, len(segment), step):
            end = start + chunk_size
            chunk = segment[start:end]

            if len(chunk) < chunk_size * .9:
                continue

            chunks.append(chunk)

    return chunks