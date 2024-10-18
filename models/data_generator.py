import numpy as np
import cv2

class DataGenerator:
    def __init__(self, face_detector, wav2lip_batch_size):
        self.face_detector = face_detector
        self.img_size = 96
        self.box = [-1, -1, -1, -1]
        self.wav2lip_batch_size = wav2lip_batch_size

    def prepare_batches(self, img_batch, mel_batch, frame_batch, coords_batch):
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, self.img_size // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        return img_batch, mel_batch, frame_batch, coords_batch

    def datagen(self, frames, mels):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if self.box[0] == -1:
            face_det_results = self.face_detector.detect_faces([frames[0]])
        else:
            print('Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = self.box
            face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

        for i, m in enumerate(mels):
            idx = 0
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()

            face = cv2.resize(face, (self.img_size, self.img_size))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= self.wav2lip_batch_size:
                yield self.prepare_batches(img_batch, mel_batch, frame_batch, coords_batch)
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            yield self.prepare_batches(img_batch, mel_batch, frame_batch, coords_batch)