"""
Talk-to-Translate 文字起こしモジュール
faster-whisper を使用して音声データからテキストを抽出します。
"""

from typing import Optional
import numpy as np
from faster_whisper import WhisperModel


def parse_language_code(ui_language_text: str) -> Optional[str]:
    """UIの言語選択肢文字列からWhisper用の言語コードへ変換"""
    if "ja" in ui_language_text:
        return "ja"
    elif "en" in ui_language_text:
        return "en"
    elif "zh" in ui_language_text:
        return "zh"
    return None  # auto


def parse_model_size(ui_model_text: str) -> str:
    """UIのモデル選択肢文字列からWhisperモデルサイズ名へ変換"""
    if "tiny" in ui_model_text:
        return "tiny"
    elif "small" in ui_model_text:
        return "small"
    return "base"


class TranscriptionEngine:
    """faster-whisper をラップした文字起こしエンジン"""

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """
        初期化
        :param model_size: "tiny", "base", "small" 等
        :param device: "cpu" (一般的なPC環境)
        :param compute_type: CPUで最速・省メモリの "int8"
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model: Optional[WhisperModel] = None
        self._load_model()

    def _load_model(self):
        """Whisperモデルをロード"""
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    def change_model(self, model_size: str):
        """モデルサイズを変更して再ロード"""
        if self.model_size != model_size:
            self.model_size = model_size
            self._load_model()

    def transcribe(self, audio_data: np.ndarray, language: Optional[str] = None) -> str:
        """
        音声データ（numpy配列, 16000Hz float32）を文字起こしする
        :param audio_data: 音声信号
        :param language: 言語コード ("ja", "en", "zh") または None (自動検出)
        :return: 文字起こしテキスト
        """
        if self.model is None or len(audio_data) == 0:
            return ""

        # 無音フィルタ（VAD）を有効化して不要な無音区間の処理をスキップ
        segments, _ = self.model.transcribe(
            audio_data,
            language=language,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            beam_size=3,
        )

        texts = [segment.text.strip() for segment in segments if segment.text.strip()]
        return " ".join(texts)
