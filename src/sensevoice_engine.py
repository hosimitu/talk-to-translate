"""
Talk-to-Translate SenseVoice 音声認識モジュール
sherpa-onnx を使用して SenseVoice-Small (ONNX int8) による超低遅延・高精度な文字起こしを提供します。
"""

import re
from collections import OrderedDict
from typing import Optional
import numpy as np
import sherpa_onnx

from src.model_integrity import download_verified
from src.runtime import (
    TRANSCRIPTION_MAX_THREADS,
    is_meaningful_transcription,
    recommend_cpu_threads,
)


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
    # 上流 main の差し替えを読み込まないよう commit SHA で固定し、取得後に SHA256 を照合する
    MODEL_REVISION = "2365baeacb507f821a0c8120fcee3d484dba7a07"
    MODEL_SHA256 = "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51"
    TOKENS_SHA256 = "f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc"

    def __init__(self, num_threads: Optional[int] = None):
        """
        初期化（モデルファイルのキャッシュ確認またはダウンロード）
        :param num_threads: CPU推論スレッド数（NoneでCPUコア数から自動決定）
        """
        self.num_threads = (
            num_threads
            if isinstance(num_threads, int) and num_threads > 0
            else recommend_cpu_threads(max_threads=TRANSCRIPTION_MAX_THREADS)
        )
        self.recognizer: Optional[sherpa_onnx.OfflineRecognizer] = None
        self._current_language = "auto"
        # 言語ごとに認識器をキャッシュする（言語切替のたびの再構築・遅延を防ぐ）。
        # モデルファイルは共有なので、認識器オブジェクト自体は軽量。
        self._recognizers: "OrderedDict[str, sherpa_onnx.OfflineRecognizer]" = OrderedDict()
        # 直近の発話から検出された言語（翻訳側のソース言語推定に利用）
        self.last_detected_language: Optional[str] = None
        self._load_model(self._current_language)

    MAX_CACHED_RECOGNIZERS = 2

    def _get_recognizer(self, language: str) -> sherpa_onnx.OfflineRecognizer:
        """指定言語の認識器を取得する（未構築なら構築し、上限超過分はLRUで破棄）"""
        if language in self._recognizers:
            self._recognizers.move_to_end(language)
            return self._recognizers[language]

        recognizer = self._build_recognizer(language)
        self._recognizers[language] = recognizer
        while len(self._recognizers) > self.MAX_CACHED_RECOGNIZERS:
            self._recognizers.popitem(last=False)
        return recognizer

    def _build_recognizer(self, language: str) -> sherpa_onnx.OfflineRecognizer:
        """固定 revision でモデルを取得・照合し、sherpa-onnx でロード"""
        model_path = download_verified(
            self.REPO_ID,
            self.MODEL_FILENAME,
            self.MODEL_REVISION,
            self.MODEL_SHA256,
        )
        tokens_path = download_verified(
            self.REPO_ID,
            self.TOKENS_FILENAME,
            self.MODEL_REVISION,
            self.TOKENS_SHA256,
        )

        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_path,
            tokens=tokens_path,
            num_threads=self.num_threads,
            use_itn=True,
            language=language if language != "auto" else "auto",
        )

    def _load_model(self, language: str):
        """下位互換用ラッパ（指定言語の認識器を構築して current に設定する）"""
        self.recognizer = self._get_recognizer(language)
        self._current_language = language

    def transcribe(self, audio_data: np.ndarray, language: Optional[str] = None) -> str:
        """
        音声データからテキストを抽出（日中英に超特化、爆速推論）
        :param audio_data: 16000Hz float32 音声波形
        :param language: 言語指定 ("ja", "zh", "en" または None)
        :return: 整形済み文字起こしテキスト
        """
        target_lang = language if language else "auto"
        recognizer = self._get_recognizer(target_lang)
        self.recognizer = recognizer
        self._current_language = target_lang

        if len(audio_data) == 0:
            return ""

        stream = recognizer.create_stream()
        # 16000Hzモノラルで波形を渡す
        stream.accept_waveform(16000, audio_data)
        recognizer.decode_stream(stream)
        raw_text = stream.result.text.strip()

        # 感情・イベントタグ (<|ja|><|NEUTRAL|><|Speech|> 等) より前に言語を取得
        self.last_detected_language = normalize_language(raw_text) or normalize_language(
            getattr(stream.result, "lang", None)
        )

        if not raw_text:
            return ""

        # SenseVoiceの感情・イベントタグを除去
        clean_text = re.sub(r"<\|.*?\|>", "", raw_text).strip()

        # 非発話音では "。" や "." など記号だけを返すことがあるため、語を含まない出力は捨てる
        if not is_meaningful_transcription(clean_text):
            return ""
        return clean_text
