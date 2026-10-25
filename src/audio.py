"""
Talk-to-Translate 音声録音モジュール
sounddevice を使用してマイクからの音声ストリームをリアルタイムにバッファリングします。
"""

import queue
import time
from typing import Optional
import numpy as np
import sounddevice as sd


class AudioRecorder:
    """マイクからの音声入力を取得・バッファリングするクラス"""

    # 推論が追いつかない場合に保持する最大チャンク数（古いものから破棄）
    MAX_QUEUED_CHUNKS = 256

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        初期化
        :param sample_rate: サンプリングレート (Whisper標準の16000Hz)
        :param channels: チャンネル数 (1: モノラル)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_queue: queue.Queue[tuple[float, np.ndarray]] = queue.Queue(
            maxsize=self.MAX_QUEUED_CHUNKS
        )
        self.dropped_chunks = 0
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
        if not self._is_recording:
            return

        # 1チャンネル分のデータをフラット化し、取得時刻とともにキューへ追加
        timestamp = time.monotonic()
        chunk = indata.copy().flatten()
        try:
            self.audio_queue.put_nowait((timestamp, chunk))
        except queue.Full:
            # 推論が追いつかず遅延が蓄積する場合は、最も古いチャンクを捨てる
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            self.dropped_chunks += 1
            try:
                self.audio_queue.put_nowait((timestamp, chunk))
            except queue.Full:
                pass

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
        self.dropped_chunks = 0
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

    def get_chunk_with_timestamp(self, timeout: float = 0.2) -> tuple[Optional[float], Optional[np.ndarray]]:
        """
        キューから音声チャンクを取得時刻（monotonic）とともに取り出す
        :param timeout: タイムアウト（秒）
        :return: (取得時刻, 音声データ)。取得できない場合は (None, None)
        """
        try:
            item = self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None

        # 後方互換: 過去形式（ndarrayのみ）で投入された場合は現在時刻を付与
        if isinstance(item, tuple) and len(item) == 2:
            return item
        return time.monotonic(), item

    def get_chunk(self, timeout: float = 0.2) -> Optional[np.ndarray]:
        """
        キューから音声データチャンクを1つ取り出す
        :param timeout: タイムアウト（秒）
        :return: 音声データ (numpy配列) または None
        """
        _, chunk = self.get_chunk_with_timestamp(timeout=timeout)
        return chunk
