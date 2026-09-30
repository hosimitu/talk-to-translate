"""
Talk-to-Translate 音声処理ワーカーモジュール
発話と無音を検出し、必要に応じてMossFormerによる音源分離（2人同時発話の個別抽出）を行い、文字起こしを実行します。
"""

import threading
import time
from typing import Callable, Optional
import numpy as np

from src.audio import AudioRecorder
from src.transcriber import TranscriptionEngine
from src.separator import SpeechSeparator


def preprocess_audio(audio: np.ndarray, target_peak: float = 0.9, max_gain: float = 6.0) -> np.ndarray:
    """
    音声波形の前処理（DCオフセット除去＆自動音量ノーマライズ）
    :param audio: 音声信号 (float32)
    :param target_peak: 目標とするピーク音量 (0.0〜1.0)
    :param max_gain: 最大増幅率
    :return: 前処理済みの音声信号
    """
    if len(audio) == 0:
        return audio

    audio = audio - np.mean(audio)
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-4:
        return audio

    gain = min(target_peak / peak, max_gain)
    normalized = audio * gain
    return np.clip(normalized, -1.0, 1.0)


class AudioProcessor:
    """発話区間と無音を検出し、文単位で文字起こし（+音源分離）を行うワーカー"""

    def __init__(
        self,
        recorder: AudioRecorder,
        transcriber: TranscriptionEngine,
        on_transcription_callback: Callable[[str], None],
        separator: Optional[SpeechSeparator] = None,
        silence_threshold: float = 0.03,
        silence_duration_sec: float = 0.8,
        min_speech_duration_sec: float = 1.0,
        max_speech_duration_sec: float = 8.0,
    ):
        """
        初期化
        :param recorder: AudioRecorder インスタンス
        :param transcriber: TranscriptionEngine インスタンス
        :param on_transcription_callback: 文字起こし完了時に呼ばれるコールバック
        :param separator: SpeechSeparator インスタンス (任意)
        """
        self.recorder = recorder
        self.transcriber = transcriber
        self.on_transcription_callback = on_transcription_callback
        self.separator = separator
        self.enable_separation = False  # 音源分離のON/OFFフラグ

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
        pre_buffer: list[np.ndarray] = []
        max_pre_buffer_chunks = 3

        is_speaking = False
        silence_start_time: Optional[float] = None
        speech_start_time: Optional[float] = None

        while self._is_running:
            chunk = self.recorder.get_chunk(timeout=0.1)
            if chunk is None or len(chunk) == 0:
                continue

            rms = float(np.sqrt(np.mean(chunk**2)))
            is_sound = rms > self.silence_threshold

            now = time.time()

            if is_sound:
                silence_start_time = None
                if not is_speaking:
                    is_speaking = True
                    speech_start_time = now
                    speech_buffer.extend(pre_buffer)
                    pre_buffer.clear()
                speech_buffer.append(chunk)
            else:
                if is_speaking:
                    speech_buffer.append(chunk)
                    if silence_start_time is None:
                        silence_start_time = now
                    elif now - silence_start_time >= self.silence_duration_sec:
                        self._process_speech(speech_buffer, sample_rate)
                        speech_buffer.clear()
                        is_speaking = False
                        silence_start_time = None
                        speech_start_time = None
                else:
                    pre_buffer.append(chunk)
                    if len(pre_buffer) > max_pre_buffer_chunks:
                        pre_buffer.pop(0)

            if is_speaking and speech_start_time and (now - speech_start_time >= self.max_speech_duration_sec):
                self._process_speech(speech_buffer, sample_rate)
                speech_buffer.clear()
                is_speaking = False
                silence_start_time = None
                speech_start_time = None

        if speech_buffer:
            self._process_speech(speech_buffer, sample_rate)

    def _process_speech(self, buffer: list[np.ndarray], sample_rate: int):
        """蓄積された発話音声を処理（音源分離ONなら分離、OFFなら直接文字起こし）"""
        if not buffer:
            return

        raw_audio = np.concatenate(buffer)
        duration_sec = len(raw_audio) / sample_rate

        if duration_sec < self.min_speech_duration_sec:
            return

        # 1. 音量自動ノーマライズ
        audio_data = preprocess_audio(raw_audio, target_peak=0.9, max_gain=6.0)

        # 2. 音源分離が有効な場合
        if self.enable_separation and self.separator is not None:
            try:
                tracks = self.separator.separate(audio_data, sample_rate)
                if len(tracks) > 1:
                    # 2人以上の声が分離された場合
                    for i, track in enumerate(tracks):
                        norm_track = preprocess_audio(track)
                        text = self.transcriber.transcribe(norm_track, language=self._language)
                        if text and text.strip():
                            speaker_label = f"[話者{i+1}]"
                            self.on_transcription_callback(f"{speaker_label} {text.strip()}")
                    return
            except Exception as e:
                print(f"[AudioProcessor] 音源分離スキップ（フォールバック）: {e}")

        # 3. 通常の単一文字起こし（音源分離OFFまたは1人の場合）
        try:
            text = self.transcriber.transcribe(audio_data, language=self._language)
            if text and text.strip():
                self.on_transcription_callback(text.strip())
        except Exception as e:
            print(f"[AudioProcessor Error] 文字起こし処理失敗: {e}")
