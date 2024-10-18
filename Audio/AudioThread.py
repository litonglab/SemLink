import sys
import sherpa_ncnn
import sounddevice as sd
from PyQt6.QtCore import QThread, pyqtSignal
import threading

class AudioThread(QThread):
    def create_recognizer(self):
        recognizer = sherpa_ncnn.Recognizer(
            tokens="./sherpa/tokens.txt",
            encoder_param="./sherpa/encoder_jit_trace-pnnx.ncnn.param",
            encoder_bin="./sherpa/encoder_jit_trace-pnnx.ncnn.bin",
            decoder_param="./sherpa/decoder_jit_trace-pnnx.ncnn.param",
            decoder_bin="./sherpa/decoder_jit_trace-pnnx.ncnn.bin",
            joiner_param="./sherpa/joiner_jit_trace-pnnx.ncnn.param",
            joiner_bin="./sherpa/joiner_jit_trace-pnnx.ncnn.bin",
            num_threads=8,
        )
        return recognizer

    def __init__(self, socket_comm):
        super().__init__()
        self.socket_comm = socket_comm;
        self.recognizer = self.create_recognizer()
        self.sample_rate = self.recognizer.sample_rate
        self.samples_per_read = int(0.1 * self.sample_rate)  # 0.1 second = 100 ms

    def run(self):
        last_result = ""
        with sd.InputStream(
                channels=1, dtype="float32", samplerate=self.sample_rate
        ) as s:
            while True:
                samples, _ = s.read(self.samples_per_read)
                audio_data = b"Audio:" + samples.tobytes()
                self.socket_comm.send_data(audio_data, self.socket_comm.conn)
                # samples = samples.reshape(-1)
                # self.recognizer.accept_waveform(self.sample_rate, samples)
                # result = self.recognizer.text
                # if last_result != result:
                #     new_text = result[len(last_result):]  # 新的部分
                #     if new_text:
                #         self.socket_comm.send_data("Text:" + new_text, self.socket_comm.conn)
                #     last_result = result

