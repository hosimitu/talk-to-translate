"""
Talk-to-Translate 音声処理ワーカーモジュール
録音された音声を適切なチャンクに集約し、バックグラウンドで文字起こしを実行します。
"""

import threading
import time
from typing import Callable, Optional
import numpy as np

from src.audio import AudioRecorder
from src.transcriber import TranscriptionEngine


class AudioProcessor:
    """音声データを集約しバックグラウンドで文字起こしを行うワーカー"""

    def __init__(
        self,
        recorder: AudioRecorder,
        transcriber: TranscriptionEngine,
        on_transcription_callback: Callable[[str], None],
        chunk_duration_sec: float = 3.0,
    ):
        """
        初期化
        :param recorder: AudioRecorder インスタンス
        :param transcriber: TranscriptionEngine インスタンス
        :param on_transcription_callback: 文字起こし完了時に呼ばれるコールバック (引数: テキスト)
        :param chunk_duration_sec: 1回の文字起こしに渡す音声の長さ（秒）
        """
        self.recorder = recorder
        self.transcriber = transcriber
        self.on_transcription_callback = on_transcription_callback
        self.chunk_duration_sec = chunk_duration_sec

        self._thread: Optional[threading.Thread] = None
        self._is_running = False
        self._language: Optional[str] = None

    @property
    def is_running(self) -> bool:
        return self._is_running

    def start(self, language: Optional[str] = None):
        """ワーカーを開始"""
        if self._is_running:
            return

        self._is_running = True
        self._language = language
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """ワーカーを停止"""
        self._is_running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _worker_loop(self):
        """音声収集と文字起こしのワーカーループ"""
        sample_rate = self.recorder.sample_rate
        target_samples = int(self.chunk_duration_sec * sample_rate)
        buffer: list[np.ndarray] = []
        current_samples = 0

        while self._is_running:
            chunk = self.recorder.get_chunk(timeout=0.2)
            if chunk is not None:
                buffer.append(chunk)
                current_samples += len(chunk)

            # 設定したチャンク長に達したら文字起こしを実行
            if current_samples >= target_samples:
                audio_data = np.concatenate(buffer)
                buffer = []
                current_samples = 0
                self._run_transcription(audio_data)

        # 停止時にバッファに残っている音声（最低0.8秒以上あれば）を最後に処理
        if current_samples >= int(sample_rate * 0.8):
            audio_data = np.concatenate(buffer)
            self._run_transcription(audio_data)

    def _run_transcription(self, audio_data: np.ndarray):
        """文字起こしを実行しコールバックを呼び出す"""
        try:
            text = self.transcriber.transcribe(audio_data, language=self._language)
            if text and text.strip():
                self.on_transcription_callback(text.strip())
        except Exception as e:
            print(f"[AudioProcessor Error] 文字起こし処理失敗: {e}")
