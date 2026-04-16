import os
import torch
import tempfile
import numpy as np
import soundfile as sf
import librosa
import whisper

TARGET_SR = 22050  # Frequência padrão do XTTS

class EmbeddingService:
    def __init__(self, tts_model):
        """
        Responsável por:
        - gerar embeddings de speaker
        - cachear embeddings
        - selecionar melhores chunks
        """
        self.tts_model = tts_model
        self.cache = {}
        self.asr_cache = {}
        self.asr = whisper.load_model("base")

        os.makedirs("embeddings", exist_ok=True)

    def fast_score(self, chunk):
        """
        Pré-filtro rápido:
        - energia * variação
        """
        return np.mean(np.abs(chunk)) * np.std(chunk)

    def score_chunk(self, chunk):
        """
        Score completo:
        - ASR
        - energia
        - pitch
        """

        # descarta chunks muito pequenos
        if len(chunk) < 16000:
            return 0

        key = hash(chunk.tobytes())

        # cache ASR
        if key in self.asr_cache:
            text = self.asr_cache[key]
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, chunk, TARGET_SR)

                try:
                    result = self.asr.transcribe(tmp.name, fp16=False)
                    text = result.get("text", "").strip()
                    self.asr_cache[key] = text
                except:
                    return 0

        # descarta fala ruim (muito curta)
        if len(text) < 5:
            return 0

        energy = np.mean(np.abs(chunk))
        std = np.std(chunk)

        if std < 0.01 or energy < 0.05:
            return 0

        pitches, _ = librosa.piptrack(y=chunk, sr=TARGET_SR)
        pitch_var = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0

        return (energy * std * 10) + (pitch_var * 0.5) + (len(text) * 2)

    def get_or_create(self, speaker, audio_chunks):
        """
        Busca embedding:
        1. cache
        2. disco
        3. cria novo
        """
        if speaker in self.cache:
            return self.cache[speaker]

        path = f"embeddings/{speaker}.pt"

        if os.path.exists(path):
            data = torch.load(path)
            self.cache[speaker] = data
            return data

        embedding = self.create_embedding(audio_chunks)
        torch.save(embedding, path)

        self.cache[speaker] = embedding
        return embedding

    def create_embedding(self, audio_chunks):
        """
        Gera embedding de speaker a partir de múltiplos chunks de áudio.

        Pipeline:
        1. Seleciona melhores chunks 
        2. Extrai embeddings individuais 
        3. Faz agregação (mediana)
        """

        # PRÉ-SELEÇÃO 
        # Usa um score barato  para filtrar rapidamente
        # Evita rodar ASR em todos os chunks 
        scored = sorted(
            [(self.fast_score(c), c) for c in audio_chunks],
            reverse=True
        )[:15]  # pega top 15 candidatos


        # SELEÇÃO REFINADA
        scored = sorted(
            [(self.score_chunk(c), c) for _, c in scored],
            reverse=True
        )[:8]  # pega top 8 melhores chunks


        gpt_list = []        # contexto de linguagem 
        speaker_list = []    # identidade da voz


        # EXTRAÇÃO DE EMBEDDINGS
        for _, chunk in scored:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, chunk, TARGET_SR)

                # Extrai embeddings do modelo
                gpt, speaker = self.tts_model.get_conditioning_latents(
                    audio_path=tmp.name
                )

                # Armazena embeddings individuais
                gpt_list.append(gpt)
                speaker_list.append(speaker)

        # AGREGAÇÃO
        # Usa MEDIANA ao invés de média:
        # - mais resistente a outliers
        # - evita chunks ruins influenciarem o resultado final
        return {
            "gpt_cond_latent": torch.median(
                torch.stack(gpt_list), dim=0
            ).values,

            "speaker_embedding": torch.median(
                torch.stack(speaker_list), dim=0
            ).values,
        }