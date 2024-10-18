from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage
import cv2
import pickle
from time import sleep

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self, socket_comm, picture_gap):
        super().__init__()
        self.socket_comm = socket_comm;
        self.picture_gap = picture_gap

    def run(self):
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, cv_img = cap.read()
            if ret:
                # 将图像转换为QImage对象
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                p = convert_to_Qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                self.change_pixmap_signal.emit(p)

                _, buffer = cv2.imencode('.jpg', cv_img)
                image_data = b"Image:" + buffer.tobytes()
                self.socket_comm.send_data(image_data, self.socket_comm.conn)
            else:
                break
            # sleep(self.picture_gap)
        # 释放摄像头资源
        cap.release()
