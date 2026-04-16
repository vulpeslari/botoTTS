from TTS.api import TTS

class TTSService:
    def __init__(self):
        """
        Carrega modelo XTTS (GPU)
        """
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        self.tts.to("cuda")

    def infer(self, text, embedding):
        """
        Gera áudio a partir de:
        - texto
        - embedding de speaker
        """
        return self.tts.synthesizer.tts_model.inference(
            text=text,
            language="pt",
            gpt_cond_latent=embedding["gpt_cond_latent"],
            speaker_embedding=embedding["speaker_embedding"],
            temperature=0.65,
            repetition_penalty=10.0,
            top_k=50,
            top_p=0.85,
            do_sample=True,
            speed=1.05,
        )