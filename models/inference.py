import numpy as np
import cv2
import torch, face_detection

from tqdm import tqdm
from PyQt6.QtGui import QImage
from PyQt6.QtCore import QObject, pyqtSignal
from models import Wav2Lip
from .data_generator import DataGenerator
from .face_detector import FaceDetector
from Util import audio

class Wav2LipModel:
    def __init__(self, checkpoint_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} for inference.'.format(self.device))
        self.model = self.load_model(checkpoint_path)
        print("Model loaded")

    def _load_checkpoint(self, checkpoint_path):
        if self.device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        return checkpoint

    def load_model(self, path):
        model = Wav2Lip()
        print("Load checkpoint from: {}".format(path))
        checkpoint = self._load_checkpoint(path)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)

        model = model.to(self.device)
        return model.eval()

    def infer(self, mel_batch, img_batch):
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

        with torch.no_grad():
            pred = self.model(mel_batch, img_batch)

        return pred.cpu().numpy().transpose(0, 2, 3, 1) * 255

class Inference(QObject):
    change_pixmap_signal = pyqtSignal(QImage)
    def __init__(self, checkpoint_path):
        super().__init__()
        self.wav2lip_batch_size = 128
        self.face_detector = FaceDetector('cuda' if torch.cuda.is_available() else 'cpu')
        self.wav2lip_model = Wav2LipModel(checkpoint_path)
        self.data_generator = DataGenerator(self.face_detector, self.wav2lip_batch_size)

    def inference(self, full_frames, mels, fps):
        batch_size = self.wav2lip_batch_size
        gen = self.data_generator.datagen(full_frames.copy(), mels)

        for i, (img_batch, mel_batch, frames, coords) in enumerate(
                tqdm(gen, total=int(np.ceil(float(len(mels)) / batch_size)))):
            pred = self.wav2lip_model.infer(mel_batch, img_batch)

            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p

                # Convert frame to QImage for displaying in QLabel
                rgb_image = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

                # Emit the signal to update the QLabel with the new image
                self.change_pixmap_signal.emit(qt_image)

                if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                    break