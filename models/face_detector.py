import face_detection
import numpy as np
from tqdm import tqdm

class FaceDetector:
    def __init__(self, device, batch_size=16, pads=[0, 20, 0, 0], nosmooth=True):
        self.device = device
        self.batch_size = batch_size
        self.pads = pads
        self.nosmooth = nosmooth
        self.detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=self.device)

    def get_smoothened_boxes(self, boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i: i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def detect_faces(self, images):
        while True:
            predictions = []
            try:
                for i in tqdm(range(0, len(images), self.batch_size)):
                    predictions.extend(self.detector.get_detections_for_batch(np.array(images[i:i + self.batch_size])))
            except RuntimeError:
                if self.batch_size == 1:
                    raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
                self.batch_size //= 2
                print('Recovering from OOM error; New batch size: {}'.format(self.batch_size))
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = self.pads
        for rect, image in zip(predictions, images):
            if rect is None:
                cv2.imwrite('temp/faulty_frame.jpg', image)  # check this frame where the face was not detected.
                raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)

            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not self.nosmooth:
            boxes = self.get_smoothened_boxes(boxes, T=5)
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

        return results