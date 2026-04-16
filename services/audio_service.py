from utils.audio_utils import convert, split_audio, audio_process, audio_process_output

class AudioService:
    def preprocess(self, audio_input):
        audio, sr = convert(audio_input)
        chunks = split_audio(audio, sr)

        processed = []
        for chunk in chunks:
            p = audio_process(chunk, sr)
            if p is not None:
                processed.append(p)

        return processed

    def postprocess(self, wav):
        return audio_process_output(wav, 22050)