import socket
import pickle
import struct
import numpy as np
import cv2
import threading

from PyQt6.QtCore import Qt, QThread
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtGui import QImage
from models.aigc import Wav2LipThread

class SocketCommunicator(QObject):
    log_text_signal = pyqtSignal(str)
    update_text_signal = pyqtSignal(str)
    change_pixmap_signal = pyqtSignal(QImage)
    confirm_signal = pyqtSignal(tuple)
    def __init__(self, host, port, wav2lip_model):
        super().__init__()
        self.host = host
        self.port = port
        self.wav2lip_model = wav2lip_model
        self.server_sock = None
        self.client_sock = None
        self.conn = None
        self.addr = None
        self.audio_buffer = None
        self.listening = False
        self.confirm_event = threading.Event()
        self._initialize_sockets()

    def _initialize_sockets(self):
        # Initialize sockets
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Set socket options to allow quick reuse and prevent TIME_WAIT issues
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def set_address(self, ip, port):
        self.host = ip
        self.port = int(port)

    def start_server(self):
        if self.listening or self.conn:
            if self.listening:
                self.log_text_signal.emit(f"[INFO] Listening on {self.host}:{self.port}...\n")
            else:
                self.log_text_signal.emit(f"[INFO] Connected!\n")
            return False

        try:
            if self.server_sock.fileno() == -1:
                self._initialize_sockets()
            self.server_sock.bind((self.host, self.port))
            self.port = self.server_sock.getsockname()[1]
            self.log_text_signal.emit(f"[INFO] Listening on {self.host}:{self.port}...\n")
            self.server_sock.listen(1)
            self.listening = True

            while True:
                self.log_text_signal.emit(f"[INFO] Waiting for connection...\n")
                # 1. 等待连接请求
                potential_conn, potential_addr = self.server_sock.accept()

                # 2. 请求用户确认连接
                self.log_text_signal.emit(
                    f"[INFO] Incoming connection from {potential_addr}, awaiting confirmation...\n")

                self.confirm_signal.emit(potential_addr)
                # 等待确认结果
                self.confirm_event.wait()

                if self.confirm_result:
                    self.conn, self.addr = potential_conn, potential_addr
                    self.listening = False
                    self.log_text_signal.emit(f"[INFO] Connected by {self.addr}\n")
                    self.receive_data()
                else:
                    self.log_text_signal.emit(f"[INFO] Connection from {potential_addr} rejected\n")
                    potential_conn.close()


        except socket.error as e:
            self.log_text_signal.emit(f"[ERROR] Server Socket Error: {e}\n")
            self.close_connection()

    def set_confirmation(self, result):
        # 主线程中调用的确认函数
        self.confirm_result = result
        self.confirm_event.set()  # 通知子线程继续执行

    def start_client(self):
        try:
            self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_sock.connect((self.host, self.port))
            self.log_text_signal.emit(f"[INFO] Connected to {self.host}:{self.port}\n")
            self.conn = self.client_sock
            self.receive_data()
        except socket.error as e:
            self.log_text_signal.emit(f"[ERROR] Connection failed: {e}\n")
            self.close_connection()

    def send_data(self, data, conn):
        serialized_data = pickle.dumps(data)
        data_length = struct.pack('!I', len(serialized_data))
        if conn:
            try:
                self.conn.sendall(data_length + serialized_data)
            except socket.error as e:
                self.log_text_signal.emit(f"[ERROR] {e}\n")
                self.close_connection()

    def receive_data(self):
        while self.conn:
            data_length = self.recvall(4)
            if not data_length:
                break
            data_length = struct.unpack('!I', data_length)[0]
            data = self.recvall(data_length)
            if data:
                deserialized_data = pickle.loads(data)
                if isinstance(deserialized_data, str) and deserialized_data.startswith("Text:"): # receive text
                    text_data = deserialized_data[len("Text:") :]
                    self.update_text_signal.emit(f"[TEXT] Received text: {text_data}\n")
                elif isinstance(deserialized_data, bytes) and deserialized_data.startswith(b"Image:"): # receive image
                    image_data = deserialized_data[len("Image:") :]
                    self.update_text_signal.emit(f"[IMAGE] Received image, audio buffer {len(self.audio_buffer)} bytes\n")
                    nparr = np.frombuffer(image_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb_image.shape
                        bytes_per_line = ch * w
                        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                        scaled_image = qt_image.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)

                        if self.audio_buffer:
                            # start Wa2Lip Thread
                            wav2Lip_thread = Wav2LipThread(scaled_image, self.audio_buffer, self.wav2lip_model)
                            wav2Lip_thread.log_text_signal.connect(self.update_text_signal.emit)
                            wav2Lip_thread.start()
                            wav2Lip_thread.wait()

                            self.audio_buffer = None
                        # self.change_pixmap_signal.emit(scaled_image)
                elif isinstance(deserialized_data, bytes) and deserialized_data.startswith(b"Audio:"): # receive audio
                    audio_data = deserialized_data[len("Audio:"):]
                    self.audio_buffer = self.audio_buffer + audio_data if self.audio_buffer else audio_data
                    audio_data_length = len(audio_data)
                    self.update_text_signal.emit(f"[AUDIO] Received audio data {audio_data_length} bytes\n")
                else:
                    self.update_text_signal.emit(f"[UNKNOWN] Received unknown data type\n")
        self.close_connection()

    def recvall(self, n):
        data = b''
        while len(data) < n:
            try:
                packet = self.conn.recv(n - len(data))
                if not packet:
                    return None
                data += packet
            except socket.error as e:
                self.log_text_signal.emit(f"[ERROR] recv failed: {e}\n")
                return None
        return data

    def close_connection(self):
        if self.conn:
            self.log_text_signal.emit(f"[INFO] Connection is closed.\n")
            try:
                self.conn.close()
            except socket.error as e:
                self.log_text_signal.emit(f"[ERROR] Closing connection failed: {e}\n")
            finally:
                self.conn = None

        if self.client_sock:
            try:
                self.client_sock.close()
            except socket.error as e:
                self.log_text_signal.emit(f"[ERROR] Closing client socket failed: {e}\n")
            finally:
                self.client_sock = None