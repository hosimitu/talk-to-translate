"""
Talk-to-Translate 音声録音モジュール
sounddevice を使用してマイクからの音声ストリームをリアルタイムにバッファリングします。
"""

import queue
from typing import Optional
import numpy as np
import sounddevice as sd


class AudioRecorder:
    """マイクからの音声入力を取得・バッファリングするクラス"""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        初期化
        :param sample_rate: サンプリングレート (Whisper標準の16000Hz)
        :param channels: チャンネル数 (1: モノラル)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._is_recording = False

    @property
    def is_recording(self) -> bool:
        """現在録音中かどうか"""
        return self._is_recording

    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        """sounddeviceのストリームから音声データを受け取るコールバック"""
        if status:
            # オーバーフロー等の警告は無視するかログ記録
            pass
        if self._is_recording:
            # 1チャンネル分のデータをフラット化してキューに追加
            self.audio_queue.put(indata.copy().flatten())

    def start(self, device_index: Optional[int] = None):
        """
        録音を開始する
        :param device_index: 使用するマイクのデバイスID (Noneの場合はデフォルト)
        """
        if self._is_recording:
            return

        # 既存キューをクリア
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        self._is_recording = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            device=device_index,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        """録音を停止する"""
        if not self._is_recording:
            return

        self._is_recording = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_chunk(self, timeout: float = 0.2) -> Optional[np.ndarray]:
        """
        キューから音声データチャンクを1つ取り出す
        :param timeout: タイムアウト（秒）
        :return: 音声データ (numpy配列) または None
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
