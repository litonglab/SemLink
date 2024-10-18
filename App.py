from PyQt6 import QtWidgets, uic
from PyQt6.QtGui import QPixmap, QTextCursor
from PyQt6.QtWidgets import QMessageBox
from models.inference import Inference

from Video.VideoThread import VideoThread
from Audio.AudioThread import AudioThread
from Socket.SocketCommunicator import SocketCommunicator
from Socket.ServerThread import ServerThread
from Socket.ClientThread import ClientThread


class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # 加载.ui文件
        uic.loadUi('UI\\App.ui', self)

        # 确保 image_label 是在 UI Designer 中 QLabel 的 objectName
        self.image_label = self.findChild(QtWidgets.QLabel, 'label')
        self.text_edit = self.findChild(QtWidgets.QTextEdit, 'textEdit')
        self.log_edit = self.findChild(QtWidgets.QTextEdit, 'LogText')
        self.call_button = self.findChild(QtWidgets.QPushButton, 'CallButton')
        self.stop_button = self.findChild(QtWidgets.QPushButton, 'StopButton')
        self.call_ip_address = self.findChild(QtWidgets.QLineEdit, 'IPInput')
        self.call_port_address = self.findChild(QtWidgets.QLineEdit, 'PortInput')

        self.wav2lip_model = Inference(checkpoint_path='./checkpoints/wav2lip_gan.pth')
        self.wav2lip_model.change_pixmap_signal.connect(self.update_image)

        self.socket_comm = SocketCommunicator('127.0.0.1', 0, self.wav2lip_model)
        self.socket_comm.log_text_signal.connect(self.update_log)
        self.socket_comm.update_text_signal.connect(self.update_text)
        self.socket_comm.change_pixmap_signal.connect(self.update_image)
        self.socket_comm.confirm_signal.connect(self.show_confirmation_dialog)

        self.call_button.clicked.connect(self.CallButtonClicked)
        self.stop_button.clicked.connect(self.StopButtonClicked)

        # 设置视频线程
        self.video_thread = VideoThread(self.socket_comm, 5)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.start()

        # 设置音频线程
        self.audio_thread = AudioThread(self.socket_comm)
        self.audio_thread.start()

        # 设置服务端通信线程
        self.server_thread = ServerThread(self.socket_comm)
        self.server_thread.start()

    def update_image(self, cv_img):
        # 转换 QImage 到 QPixmap 并更新 QLabel
        pixmap = QPixmap.fromImage(cv_img)
        self.image_label.setPixmap(pixmap)

    def update_text(self, text):
        self.text_edit.moveCursor(QTextCursor.MoveOperation.End)
        self.text_edit.insertPlainText(text)

    def update_log(self, text):
        self.log_edit.moveCursor(QTextCursor.MoveOperation.End)
        self.log_edit.insertPlainText(text)

    def CallButtonClicked(self):
        self.socket_comm.set_address(self.call_ip_address.text(), self.call_port_address.text())
        self.client_thread = ClientThread(self.socket_comm)
        self.client_thread.start()

    def show_confirmation_dialog(self, addr):
        # 在主线程中显示确认对话框
        result = QMessageBox.question(self, "Confirm Connection",
                                      f"Accept connection from {addr}?",
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        confirmation = (result == QMessageBox.StandardButton.Yes)
        # 将确认结果发送回线程
        self.socket_comm.set_confirmation(confirmation)

    def StopButtonClicked(self):
        self.socket_comm.send_data(f'\n[INFO] The other party disconnected\n', self.socket_comm.conn);
        exit(0)

    def closeEvent(self):
        self.socket_comm.close_connection()
        self.video_thread.quit()
        self.video_thread.wait()
        self.audio_thread.quit()
