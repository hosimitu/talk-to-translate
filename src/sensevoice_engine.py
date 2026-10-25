"""
Talk-to-Translate SenseVoice 音声認識モジュール
sherpa-onnx を使用して SenseVoice-Small (ONNX int8) による超低遅延・高精度な文字起こしを提供します。
"""

import re
from typing import Optional
import huggingface_hub
import numpy as np
import sherpa_onnx


# SenseVoice が返す言語表記（<|ja|>, ja, zh-CN など）を翻訳用コードへ対応付ける
_LANGUAGE_TAG_PATTERN = re.compile(r"<\|([a-zA-Z]{2,4})\|>")
_LANGUAGE_CODE_MAP = {
    "ja": "ja",
    "en": "en",
    "zh": "zh-CN",
    "ko": "ko",
    "yue": "yue",
}


def normalize_language(value: object) -> Optional[str]:
    """言語タグ・言語コードを翻訳用コードへ正規化する（不明・auto は None）"""
    if not isinstance(value, str) or not value:
        return None

    tag = _LANGUAGE_TAG_PATTERN.search(value)
    if tag:
        code = tag.group(1).lower()
    else:
        # "zh-CN" のような地域付きコードは主要部のみを見る（本文の部分一致は拾わない）
        code = value.strip().lower().split("-")[0]

    return _LANGUAGE_CODE_MAP.get(code)


class SenseVoiceEngine:
    """sherpa-onnx を用いた SenseVoice-Small 音声認識エンジン"""

    REPO_ID = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    MODEL_FILENAME = "model.int8.onnx"
    TOKENS_FILENAME = "tokens.txt"

    def __init__(self, num_threads: int = 2):
        """
        初期化（モデルファイルのキャッシュ確認またはダウンロード）
        :param num_threads: CPU推論スレッド数
        """
        self.num_threads = num_threads
        self.recognizer: Optional[sherpa_onnx.OfflineRecognizer] = None
        self._current_language = "auto"
        # 直近の発話から検出された言語（翻訳側のソース言語推定に利用）
        self.last_detected_language: Optional[str] = None
        self._load_model(self._current_language)

    def _load_model(self, language: str):
        """HuggingFace Hub からモデルを取得し sherpa-onnx でロード"""
        model_path = huggingface_hub.hf_hub_download(
            repo_id=self.REPO_ID,
            filename=self.MODEL_FILENAME,
        )
        tokens_path = huggingface_hub.hf_hub_download(
            repo_id=self.REPO_ID,
            filename=self.TOKENS_FILENAME,
        )

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_path,
            tokens=tokens_path,
            num_threads=self.num_threads,
            use_itn=True,
            language=language if language != "auto" else "auto",
        )
        self._current_language = language

    def transcribe(self, audio_data: np.ndarray, language: Optional[str] = None) -> str:
        """
        音声データからテキストを抽出（日中英に超特化、爆速推論）
        :param audio_data: 16000Hz float32 音声波形
        :param language: 言語指定 ("ja", "zh", "en" または None)
        :return: 整形済み文字起こしテキスト
        """
        target_lang = language if language else "auto"
        if self.recognizer is None or self._current_language != target_lang:
            self._load_model(target_lang)

        if self.recognizer is None or len(audio_data) == 0:
            return ""

        stream = self.recognizer.create_stream()
        # 16000Hzモノラルで波形を渡す
        stream.accept_waveform(16000, audio_data)
        self.recognizer.decode_stream(stream)
        raw_text = stream.result.text.strip()

        # 感情・イベントタグ (<|ja|><|NEUTRAL|><|Speech|> 等) より前に言語を取得
        self.last_detected_language = normalize_language(raw_text) or normalize_language(
            getattr(stream.result, "lang", None)
        )

        if not raw_text:
            return ""

        # SenseVoiceの感情・イベントタグを除去
        clean_text = re.sub(r"<\|.*?\|>", "", raw_text).strip()
        return clean_text
