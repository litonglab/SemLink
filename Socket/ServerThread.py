from PyQt6.QtCore import QThread
from Socket.SocketCommunicator import SocketCommunicator

class ServerThread(QThread):
    def __init__(self, socket_comm):
        super().__init__()
        self.socket_comm = socket_comm

    def run(self):
        if self.socket_comm.start_server():
            self.socket_comm.close_connection()