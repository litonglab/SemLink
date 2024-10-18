from PyQt6.QtCore import QThread

class ClientThread(QThread):
    def __init__(self, socket_comm):
        super().__init__()
        self.socket_comm = socket_comm

    def run(self):
        self.socket_comm.start_client()