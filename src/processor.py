"""
Talk-to-Translate 音声処理ワーカーモジュール
発話と無音（息継ぎ・文の切れ目）を検知し、文単位で適切に区切って文字起こしを実行します。
"""

import threading
import time
from typing import Callable, Optional
import numpy as np

from src.audio import AudioRecorder
from src.transcriber import TranscriptionEngine


class AudioProcessor:
    """発話区間と無音を検出し、文単位で文字起こしを行うワーカー"""

    def __init__(
        self,
        recorder: AudioRecorder,
        transcriber: TranscriptionEngine,
        on_transcription_callback: Callable[[str], None],
        silence_threshold: float = 0.015,
        silence_duration_sec: float = 0.8,
        min_speech_duration_sec: float = 1.0,
        max_speech_duration_sec: float = 8.0,
    ):
        """
        初期化
        :param recorder: AudioRecorder インスタンス
        :param transcriber: TranscriptionEngine インスタンス
        :param on_transcription_callback: 文字起こし完了時に呼ばれるコールバック
        :param silence_threshold: 発話検知の音量しきい値 (RMS)
        :param silence_duration_sec: 文末判定とする無音の継続時間（秒）
        :param min_speech_duration_sec: 処理対象とする最小音声長（秒、ノイズ除外用）
        :param max_speech_duration_sec: 強制的に区切る最大音声長（秒、遅延防止用）
        """
        self.recorder = recorder
        self.transcriber = transcriber
        self.on_transcription_callback = on_transcription_callback
        self.silence_threshold = silence_threshold
        self.silence_duration_sec = silence_duration_sec
        self.min_speech_duration_sec = min_speech_duration_sec
        self.max_speech_duration_sec = max_speech_duration_sec

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
        """音声ストリームから発話区間を検出し文字起こしを行うループ"""
        sample_rate = self.recorder.sample_rate

        speech_buffer: list[np.ndarray] = []
        pre_buffer: list[np.ndarray] = []  # 発話直前の音声を少し保持（子音の欠落防止）
        max_pre_buffer_chunks = 3

        is_speaking = False
        silence_start_time: Optional[float] = None
        speech_start_time: Optional[float] = None

        while self._is_running:
            chunk = self.recorder.get_chunk(timeout=0.1)
            if chunk is None or len(chunk) == 0:
                continue

            # 音声の音量（RMSエネルギー）を計算
            rms = float(np.sqrt(np.mean(chunk**2)))
            is_sound = rms > self.silence_threshold

            now = time.time()

            if is_sound:
                # 音声を検知
                silence_start_time = None
                if not is_speaking:
                    # 発話開始
                    is_speaking = True
                    speech_start_time = now
                    # 直前のバッファを含めて発話の頭切れを防止
                    speech_buffer.extend(pre_buffer)
                    pre_buffer.clear()
                speech_buffer.append(chunk)
            else:
                # 無音を検知
                if is_speaking:
                    speech_buffer.append(chunk)
                    if silence_start_time is None:
                        silence_start_time = now
                    # 無音が一定時間継続したかチェック
                    elif now - silence_start_time >= self.silence_duration_sec:
                        # 文の終わりと判定 -> 文字起こしへ送信
                        self._process_speech(speech_buffer, sample_rate)
                        speech_buffer.clear()
                        is_speaking = False
                        silence_start_time = None
                        speech_start_time = None
                else:
                    # 待機中の音声（発話直前用バッファとして保持）
                    pre_buffer.append(chunk)
                    if len(pre_buffer) > max_pre_buffer_chunks:
                        pre_buffer.pop(0)

            # 最大継続時間を超えた場合は強制的に一度区切る
            if is_speaking and speech_start_time and (now - speech_start_time >= self.max_speech_duration_sec):
                self._process_speech(speech_buffer, sample_rate)
                speech_buffer.clear()
                is_speaking = False
                silence_start_time = None
                speech_start_time = None

        # 停止時に残っている音声があれば処理
        if speech_buffer:
            self._process_speech(speech_buffer, sample_rate)

    def _process_speech(self, buffer: list[np.ndarray], sample_rate: int):
        """蓄積された発話音声が十分な長さであれば文字起こしを実行"""
        if not buffer:
            return

        audio_data = np.concatenate(buffer)
        duration_sec = len(audio_data) / sample_rate

        # 短すぎるノイズ（咳、クリック音等）は無視
        if duration_sec < self.min_speech_duration_sec:
            return

        try:
            text = self.transcriber.transcribe(audio_data, language=self._language)
            if text and text.strip():
                self.on_transcription_callback(text.strip())
        except Exception as e:
            print(f"[AudioProcessor Error] 文字起こし処理失敗: {e}")
