import os
import torch
import tempfile
import numpy as np
import librosa
import soundfile as sf
from TTS.api import TTS
import whisper
import time
from audio_processing import audio_process_output

TARGET_SR = 22050


class BotoTTS:
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2"):
        self.tts = TTS(model_name)
        self.embeddings_cache = {}
        self.asr_cache = {}
        
        self.tts.to("cuda")
        self.asr_model = whisper.load_model("base") 

        os.makedirs("embeddings", exist_ok=True)
        os.makedirs("voices", exist_ok=True)

    def load_speaker(self, speaker_name, audio=None):
        t = time.time()
        if speaker_name in self.embeddings_cache:
            return self.embeddings_cache[speaker_name]

        path = f"embeddings/{speaker_name}.pt"

        if not os.path.exists(path):
            self.create_embedding(speaker_name, audio)

        data = torch.load(path)
        self.embeddings_cache[speaker_name] = data
        
        print(f"TIMER LOAD SPEAKER: {time.time() - t:.4f}s")
        return data
    
    def score_chunk(self, chunk):
        if isinstance(chunk, torch.Tensor):
            chunk = chunk.cpu().numpy()

        chunk = np.asarray(chunk, dtype=np.float32)

        if len(chunk) < 16000: 
            return 0

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, chunk, TARGET_SR)

            key = hash(chunk.tobytes())

            if key in self.asr_cache:
                text = self.asr_cache[key]
            else:
                try:
                    result = self.asr_model.transcribe(tmp.name, fp16=False)
                    text = result.get("text", "").strip()
                    self.asr_cache[key] = text
                except:
                    return 0

        print(f"{text} - ")
        if len(text) < 5:
            return 0

        energy = np.mean(np.abs(chunk))
        std = np.std(chunk)
        
        if std < 0.01:
            return 0
        
        if energy < 0.05:
            return 0

        pitches, magnitudes = librosa.piptrack(y=chunk, sr=TARGET_SR)
        pitch_var = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0

        text_score = len(text)

        score = (
            (energy * std * 10) +
            (pitch_var * 0.5) +
            (text_score * 2)
        )

        return score
    
    def fast_score(self, chunk):
        energy = np.mean(np.abs(chunk))
        std = np.std(chunk)
        return energy * std

    def create_embedding(self, speaker_name, audio):
        t = time.time()
        path = f"embeddings/{speaker_name}.pt"

        if os.path.exists(path):
            return

        if not isinstance(audio, list):
            audio = [audio]

        gpt_list = []
        speaker_list = []
        
        pre_scored = [(self.fast_score(c), c) for c in audio]
        pre_scored.sort(reverse=True, key=lambda x: x[0])

        top_candidates = [c for _, c in pre_scored[:15]]

        scored = [(self.score_chunk(c), c) for c in top_candidates]
        scored.sort(reverse=True, key=lambda x: x[0])

        top_k = 8
        best_chunks = [c for _, c in scored[:top_k]]

        for chunk in best_chunks:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:

                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.cpu().numpy()

                chunk = np.asarray(chunk, dtype=np.float32)

                if len(chunk.shape) == 0 or len(chunk) < 1000:
                    continue

                sf.write(tmp.name, chunk, TARGET_SR, format="WAV")

                gpt_cond_latent, speaker_embedding = (
                    self.tts.synthesizer.tts_model.get_conditioning_latents(
                        audio_path=tmp.name
                    )
                )

                gpt_list.append(gpt_cond_latent)
                speaker_list.append(speaker_embedding)

        if len(gpt_list) == 0:
            raise ValueError("Nenhum chunk válido para gerar embedding.")

        gpt_mean = torch.median(torch.stack(gpt_list), dim=0).values
        speaker_mean = torch.median(torch.stack(speaker_list), dim=0).values 
        
        torch.save(
            {
                "gpt_cond_latent": gpt_mean,
                "speaker_embedding": speaker_mean,
            },
            path,
        )
        print(f"TIMER CREATE EMBEDDING: {time.time() - t:.4f}s")

    def generate(
        self,
        text,
        speaker,
        language="pt",
        audio=None, 
    ):
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = True
        
        data = self.load_speaker(speaker, audio)
        
        t = time.time()

        out = self.tts.synthesizer.tts_model.inference(
            text=text,
            language=language,
            gpt_cond_latent=data["gpt_cond_latent"],
            speaker_embedding=data["speaker_embedding"],
            temperature=.65,         
            repetition_penalty=10.0,  
            top_k=50,
            top_p=.85,
            do_sample=True,
            speed=1.05
        )
                
        print(f"TIMER INFERENCE: {time.time() - t:.4f}s")
        t = time.time()
        
        wav = out["wav"] if isinstance(out, dict) else out
        
        wav = audio_process_output(wav, TARGET_SR)

        output_path = f"voices/{speaker}.wav"

        self.tts.synthesizer.save_wav(wav, output_path)
        print(f"TIMER GENERATE: {time.time() - t:.4f}s")

        return output_path