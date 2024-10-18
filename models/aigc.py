from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage
from .inference import Wav2LipModel
from .inference import Inference

import numpy as np
import cv2
import Util.audio as audio
import soundfile as sf

class Wav2LipThread(QThread):
    log_text_signal = pyqtSignal(str)
    def __init__(self, image_data, audio_data, model):
        super().__init__()
        self.image_data = image_data
        self.audio_data = audio_data
        self.audio_path = "temp/temp_audio.wav"
        self.model = model
        self.fps = 25

    def image_to_cv(self, qimage):
        width = qimage.width()
        height = qimage.height()

        ptr = qimage.bits()
        ptr.setsize(height * width * 3)
        arr = np.array(ptr, dtype=np.uint8).reshape((height, width, 3))

        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def run(self):
        full_frames = [self.image_to_cv(self.image_data)]

        audio_array = np.frombuffer(self.audio_data, dtype=np.float32)
        sf.write(self.audio_path, audio_array, 16000)
        wav = audio.load_wav(self.audio_path, 16000)
        mel = audio.melspectrogram(wav)
        self.log_text_signal.emit(f"[AUDIO] Mel shape: {mel.shape}\n")

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError(
                'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_chunks = []
        mel_idx_multiplier = 80. / self.fps

        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + 16 > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - 16:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + 16])
            i += 1

        self.log_text_signal.emit(f"[AUDIO] Length of mel chunks: {len(mel_chunks)}\n")
        full_frames = full_frames[:len(mel_chunks)]
        self.model.inference(full_frames, mel_chunks, self.fps)


