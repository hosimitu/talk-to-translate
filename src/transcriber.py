"""
Talk-to-Translate 文字起こしモジュール
SenseVoice (ONNX) と faster-whisper の両方をシームレスに切り替えて利用できます。
"""

from typing import Optional, Union
import numpy as np
from faster_whisper import WhisperModel
from src.sensevoice_engine import SenseVoiceEngine


def parse_language_code(ui_language_text: str) -> Optional[str]:
    """UIの言語選択肢文字列から言語コードへ変換"""
    if "ja" in ui_language_text:
        return "ja"
    elif "en" in ui_language_text:
        return "en"
    elif "zh" in ui_language_text:
        return "zh"
    return None  # auto


def parse_model_size(ui_model_text: str) -> str:
    """UIのモデル選択肢文字列から内部モデル名へ変換"""
    if "SenseVoice" in ui_model_text or "sensevoice" in ui_model_text.lower():
        return "sensevoice"
    elif "small" in ui_model_text.lower():
        return "small"
    return "large-v3-turbo"


class TranscriptionEngine:
    """SenseVoice または Whisper を統合管理する文字起こしエンジン"""

    def __init__(self, model_size: str = "sensevoice", device: str = "cpu", compute_type: str = "int8"):
        """
        初期化
        :param model_size: "sensevoice", "large-v3-turbo", "small"
        :param device: "cpu"
        :param compute_type: "int8"
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

        # エンジンのインスタンス
        self.sensevoice_engine: Optional[SenseVoiceEngine] = None
        self.whisper_model: Optional[WhisperModel] = None

        self._load_model()

    def _load_model(self):
        """選択されているモデルをロード"""
        if self.model_size == "sensevoice":
            if self.sensevoice_engine is None:
                self.sensevoice_engine = SenseVoiceEngine()
        else:
            # Whisperモデルのロード
            self.whisper_model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

    def change_model(self, model_size: str):
        """モデルを変更して再ロード"""
        if self.model_size != model_size:
            self.model_size = model_size
            self._load_model()

    def transcribe(self, audio_data: np.ndarray, language: Optional[str] = None) -> str:
        """
        音声データを文字起こしする
        :param audio_data: 16000Hz float32 音声信号
        :param language: 言語コード ("ja", "zh", "en" または None)
        :return: 文字起こしテキスト
        """
        if len(audio_data) == 0:
            return ""

        # --- 1. SenseVoice (メイン・超低遅延) ---
        if self.model_size == "sensevoice":
            if self.sensevoice_engine is None:
                self._load_model()
            return self.sensevoice_engine.transcribe(audio_data, language=language)

        # --- 2. Whisper (バックアップ・長文汎用) ---
        if self.whisper_model is None:
            self._load_model()

        initial_prompt = None
        if language == "ja":
            initial_prompt = "こんにちは。日本語の会話をリアルタイムで文字起こしします。よろしくお願いします。"
        elif language == "zh":
            initial_prompt = "你好，正在进行语音转文字。"

        segments, _ = self.whisper_model.transcribe(
            audio_data,
            language=language,
            initial_prompt=initial_prompt,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=400,
                threshold=0.3,
            ),
            beam_size=3,
            temperature=0.0,
        )

        texts = [segment.text.strip() for segment in segments if segment.text.strip()]
        return " ".join(texts)
