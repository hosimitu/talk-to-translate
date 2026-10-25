"""
Talk-to-Translate 文字起こしモジュール
SenseVoice (ONNX) と faster-whisper の両方をシームレスに切り替えて利用できます。
"""

from typing import Optional
import numpy as np
from faster_whisper import WhisperModel
from src.config import TRANSCRIPTION_MODELS
from src.sensevoice_engine import SenseVoiceEngine, normalize_language


# UIの選択肢（src/config.py）と内部モデル名の対応表
_MODEL_SIZE_BY_LABEL = dict(
    zip(TRANSCRIPTION_MODELS, ("sensevoice", "large-v3-turbo", "small"))
)


def detect_acceleration() -> tuple[str, str]:
    """
    利用可能な推論デバイスと演算精度を返す。

    CUDA対応GPUとCTranslate2のCUDAライブラリが揃っている場合のみ ("cuda", "float16")、
    それ以外（未搭載・DLL不足・判定失敗）は ("cpu", "int8") を返す。
    """
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


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
    if ui_model_text in _MODEL_SIZE_BY_LABEL:
        return _MODEL_SIZE_BY_LABEL[ui_model_text]

    # 旧表記・部分一致へのフォールバック
    lowered = str(ui_model_text).lower()
    if "sensevoice" in lowered:
        return "sensevoice"
    elif "small" in lowered:
        return "small"
    return "large-v3-turbo"


class TranscriptionEngine:
    """SenseVoice または Whisper を統合管理する文字起こしエンジン"""

    def __init__(
        self,
        model_size: str = "sensevoice",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ):
        """
        初期化
        :param model_size: "sensevoice", "large-v3-turbo", "small"
        :param device: "cpu" / "cuda"（未指定時はGPUを自動判定）
        :param compute_type: "int8" / "float16"（未指定時はdeviceに応じて自動選択）
        """
        default_device, default_compute_type = detect_acceleration()
        self.model_size = model_size
        self.device = device or default_device
        self.compute_type = compute_type or (
            default_compute_type if self.device == "cuda" else "int8"
        )

        # 直近の文字起こしで検出された言語（翻訳側のソース言語推定に利用）
        self.last_detected_language: Optional[str] = None

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
            # Whisperモデルのロード（CUDA初期化失敗時はCPUへ自動フォールバック）
            if self.whisper_model is None:
                try:
                    self.whisper_model = WhisperModel(
                        self.model_size,
                        device=self.device,
                        compute_type=self.compute_type,
                    )
                except Exception as e:
                    if self.device == "cpu":
                        raise
                    print(
                        f"[TranscriptionEngine] {self.device} でのモデル読み込みに失敗しました。"
                        f"CPU (int8) で再試行します: {e}"
                    )
                    self.device = "cpu"
                    self.compute_type = "int8"
                    self.whisper_model = WhisperModel(
                        self.model_size,
                        device="cpu",
                        compute_type="int8",
                    )

    def change_model(self, model_size: str):
        """モデルを変更して再ロード"""
        if self.model_size != model_size:
            self.model_size = model_size
            self.whisper_model = None
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
            text = self.sensevoice_engine.transcribe(audio_data, language=language)
            detected = self.sensevoice_engine.last_detected_language
            if detected:
                self.last_detected_language = detected
            return text

        # --- 2. Whisper (バックアップ・長文汎用) ---
        if self.whisper_model is None:
            self._load_model()

        # AudioProcessor側で既に発話区間（VAD）が切り出されているため、
        # Whisper側の二重VADフィルタ（vad_filter=True）は無効化（高速化・短文誤カット防止・アセット依存解消）
        # チャンクごとの独立推論を行い、プロンプト・前文起因のハルシネーション（繰り返しループ）を防ぐため
        # condition_on_previous_text=False を指定
        segments, info = self.whisper_model.transcribe(
            audio_data,
            language=language,
            condition_on_previous_text=False,
            vad_filter=False,
            beam_size=3,
            temperature=0.0,
        )

        detected = normalize_language(getattr(info, "language", None))
        if detected:
            self.last_detected_language = detected

        texts = [
            segment.text.strip()
            for segment in segments
            if segment.text.strip() and segment.no_speech_prob < 0.6
        ]
        return " ".join(texts)
