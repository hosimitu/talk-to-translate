"""
Talk-to-Translate 音声録音モジュール
sounddevice を使用してマイクからの音声ストリームをリアルタイムにバッファリングします。
"""

import queue
import time
from typing import Optional
import numpy as np
import sounddevice as sd


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """
    音声を任意のサンプリングレートへ変換する（依存追加なしの軽量実装）。

    ダウンサンプリング時は移動平均で高域を軽く減衰させてから線形補間する。
    VAD・音声認識用途では十分な品質で、16kHz非対応マイクのフォールバックに使う。
    """
    if audio is None or len(audio) == 0:
        return np.zeros(0, dtype=np.float32)
    if source_rate <= 0 or target_rate <= 0 or source_rate == target_rate:
        return audio.astype(np.float32, copy=False)

    samples = audio.astype(np.float32, copy=False)

    # エイリアシング抑制（間引き倍率ぶんの移動平均）
    if target_rate < source_rate:
        window = max(1, int(round(source_rate / target_rate)))
        if window > 1 and len(samples) >= window:
            kernel = np.ones(window, dtype=np.float32) / window
            samples = np.convolve(samples, kernel, mode="same").astype(np.float32)

    target_length = int(round(len(samples) * target_rate / source_rate))
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    if target_length == len(samples):
        return samples

    src_positions = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    dst_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(dst_positions, src_positions, samples).astype(np.float32)


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
        # 実際にマイクを開いたレート（16kHz以外ならコールバックでリサンプルする）
        self.input_sample_rate = sample_rate

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
        if self.input_sample_rate != self.sample_rate:
            chunk = resample_audio(chunk, self.input_sample_rate, self.sample_rate)
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

        self.dropped_chunks = 0
        self.input_sample_rate = self._select_input_sample_rate(device_index)
        stream = sd.InputStream(
            samplerate=self.input_sample_rate,
            channels=self.channels,
            dtype="float32",
            device=device_index,
            callback=self._callback,
        )
        self._is_recording = True
        try:
            stream.start()
        except Exception:
            self._is_recording = False
            stream.close()
            raise
        self._stream = stream

    def _select_input_sample_rate(self, device_index: Optional[int]) -> int:
        """
        マイクが16kHzを受け付けるか検証し、不可ならデバイス既定のレートを使う。
        （一部のBluetoothマイクなど、16kHz固定だとストリームを開けない場合のフォールバック）
        """
        try:
            sd.check_input_settings(
                device=device_index,
                channels=self.channels,
                samplerate=self.sample_rate,
                dtype="float32",
            )
            return self.sample_rate
        except Exception as e:
            try:
                info = (
                    sd.query_devices(device_index, "input")
                    if device_index is not None
                    else sd.query_devices(kind="input")
                )
                fallback_rate = int(round(info.get("default_samplerate") or 0))
            except Exception:
                fallback_rate = 0

            if fallback_rate <= 0 or fallback_rate == self.sample_rate:
                # フォールバック先も決められない場合は、元のエラーをそのまま伝える
                raise e

            print(
                f"[AudioRecorder] {self.sample_rate}Hzでマイクを開けないため、"
                f"{fallback_rate}Hzで録音し 16kHz に変換します: {e}"
            )
            return fallback_rate

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
