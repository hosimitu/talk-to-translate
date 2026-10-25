"""
Talk-to-Translate 音声処理ワーカーモジュール
発話と無音を検出し、必要に応じて話者識別（3D-speaker ONNX）を行い、文字起こしを実行します。
"""

import threading
import time
from typing import Callable, Optional
import numpy as np

from src.audio import AudioRecorder
from src.transcriber import TranscriptionEngine
from src.diarizer import SpeakerDiarizer


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


def compute_effective_threshold(silence_threshold: float, noise_floor: float) -> float:
    """
    環境音に応じた有音判定の閾値を計算する。

    下限は 0.003、上限は基準閾値の3倍まで許容する。上限を固定値にすると
    騒がしい環境でノイズフロアが上がっても閾値が追従できず、無音を検出できなくなる。
    """
    return max(0.003, min(silence_threshold * 3.0, noise_floor * 2.5))


class AudioProcessor:
    """発話区間と無音を検出し、文単位で文字起こし（+話者識別）を行うワーカー"""

    def __init__(
        self,
        recorder: AudioRecorder,
        transcriber: TranscriptionEngine,
        on_transcription_callback: Callable[[str], None],
        diarizer: Optional[SpeakerDiarizer] = None,
        silence_threshold: float = 0.006,
        silence_duration_sec: float = 0.8,
        min_speech_duration_sec: float = 0.3,
        max_speech_duration_sec: float = 8.0,
        on_activity_callback: Optional[Callable[[float, bool], None]] = None,
    ):
        """
        初期化
        :param recorder: AudioRecorder インスタンス
        :param transcriber: TranscriptionEngine インスタンス
        :param on_transcription_callback: 文字起こし完了時に呼ばれるコールバック
        :param diarizer: SpeakerDiarizer インスタンス (推奨・多話者識別)
        :param silence_threshold: 有音判定の基準RMS値 (デフォルト0.006)
        :param silence_duration_sec: 発話終了とみなす無音継続時間 (秒)
        :param min_speech_duration_sec: 処理対象とする最小発話時間 (秒、短文相槌に対応)
        :param max_speech_duration_sec: 1回の処理に含める最大発話時間 (秒)
        :param on_activity_callback: 音声レベルおよび発話状態更新コールバック (rms: float, is_active: bool)
        """
        self.recorder = recorder
        self.transcriber = transcriber
        self.on_transcription_callback = on_transcription_callback
        self.diarizer = diarizer
        self.enable_diarization = False  # 話者識別のON/OFFフラグ
        self.on_activity_callback = on_activity_callback

        self.silence_threshold = silence_threshold
        self.silence_duration_sec = silence_duration_sec
        self.min_speech_duration_sec = min_speech_duration_sec
        self.max_speech_duration_sec = max_speech_duration_sec
        self.noise_floor = 0.002  # バックグラウンド環境音の初期推定値

        self._thread: Optional[threading.Thread] = None
        self._is_running = False
        self._language: Optional[str] = None
        self._last_dropped_chunks = 0

    @property
    def is_running(self) -> bool:
        return self._is_running

    def start(self, language: Optional[str] = None):
        """ワーカーを開始"""
        if self._is_running:
            return

        self._is_running = True
        self._language = language
        self._last_dropped_chunks = 0
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """ワーカーを停止"""
        self._is_running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _next_chunk(self) -> tuple[Optional[float], Optional[np.ndarray]]:
        """録音側から音声チャンクと取得時刻を取得する（旧Recorder実装にも対応）"""
        getter = getattr(self.recorder, "get_chunk_with_timestamp", None)
        if callable(getter):
            return getter(timeout=0.1)

        chunk = self.recorder.get_chunk(timeout=0.1)
        return (time.monotonic() if chunk is not None else None), chunk

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
            chunk_time, chunk = self._next_chunk()
            if chunk is None or len(chunk) == 0:
                continue

            # 推論が追いつかず音声が破棄された場合は、50チャンクごとに通知
            dropped = getattr(self.recorder, "dropped_chunks", 0)
            if dropped >= self._last_dropped_chunks + 50:
                print(
                    f"[AudioProcessor] 音声処理が追いつかず古い音声を破棄しました "
                    f"(累計 {dropped} チャンク)"
                )
                self._last_dropped_chunks = dropped

            # 取得時刻（録音時点）を基準に発話・無音の継続時間を判定する。
            # 推論で処理が遅延しても、実際の音声上の時間関係が崩れない。
            now = chunk_time if chunk_time is not None else time.monotonic()
            rms = float(np.sqrt(np.mean(chunk**2)))

            # 動的ノイズフロアの追従更新（非発話時のみ。突発音の影響を抑えるため上限を設ける）
            if not is_speaking:
                self.noise_floor = 0.95 * self.noise_floor + 0.05 * min(
                    rms, self.silence_threshold * 4.0
                )

            # 実環境適応の有効閾値 (低感度マイクと高ノイズ環境の両方に対応)
            effective_threshold = compute_effective_threshold(
                self.silence_threshold, self.noise_floor
            )
            is_sound = rms > effective_threshold

            # UI側の音量メーターや発話状態表示へ通知
            if self.on_activity_callback:
                try:
                    self.on_activity_callback(rms, is_speaking or is_sound)
                except Exception:
                    pass

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
        """蓄積された発話音声を処理（話者識別ONなら[話者N]タグを付与、OFFなら直接文字起こし）"""
        if not buffer:
            return

        raw_audio = np.concatenate(buffer)
        duration_sec = len(raw_audio) / sample_rate

        if duration_sec < self.min_speech_duration_sec:
            return

        # 1. 音量自動ノーマライズ
        audio_data = preprocess_audio(raw_audio, target_peak=0.9, max_gain=6.0)

        # 2. 話者識別（ダイアライザーが有効な場合）
        speaker_tag = ""
        if self.enable_diarization and self.diarizer is not None:
            try:
                spk = self.diarizer.identify_speaker(audio_data, sample_rate)
                if spk:
                    speaker_tag = f"[{spk}] "
            except Exception as e:
                print(f"[AudioProcessor] 話者識別スキップ（フォールバック）: {e}")

        # 3. 通常の文字起こし（話者タグ付き または タグなし）
        try:
            text = self.transcriber.transcribe(audio_data, language=self._language)
            if text and text.strip():
                final_text = f"{speaker_tag}{text.strip()}"
                self.on_transcription_callback(final_text)
        except Exception as e:
            print(f"[AudioProcessor Error] 文字起こし処理失敗: {e}")
            import traceback
            traceback.print_exc()
