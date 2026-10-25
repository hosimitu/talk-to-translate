import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.transcriber import parse_language_code, parse_model_size, TranscriptionEngine
from src.sensevoice_engine import normalize_language


def test_parsers():
    assert parse_language_code("自動検出 (auto)") is None
    assert parse_language_code("日本語 (ja)") == "ja"
    assert parse_language_code("英語 (en)") == "en"
    assert parse_language_code("中国語 (zh)") == "zh"

    assert parse_model_size("SenseVoice-Small (超高速・日中英推奨)") == "sensevoice"
    assert parse_model_size("Whisper: large-v3-turbo (高精度)") == "large-v3-turbo"
    assert parse_model_size("Whisper: small (軽量)") == "small"

    # SenseVoice の言語タグ抽出（翻訳側のソース言語推定に使用）
    assert normalize_language("<|ja|><|NEUTRAL|><|Speech|>こんにちは") == "ja"
    assert normalize_language("<|zh|><|NEUTRAL|>你好") == "zh-CN"
    assert normalize_language("<|en|>hello") == "en"
    assert normalize_language("<|ko|>안녕") == "ko"
    assert normalize_language("zh-CN") == "zh-CN"
    assert normalize_language("auto") is None
    assert normalize_language("") is None
    assert normalize_language(None) is None
    print("[OK] パース関数のテスト成功")


def test_sensevoice_engine():
    print("SenseVoiceエンジンのテスト実行中...")
    engine = TranscriptionEngine(model_size="sensevoice")
    dummy_audio = np.zeros(16000, dtype=np.float32)
    res = engine.transcribe(dummy_audio, language="ja")
    assert isinstance(res, str)
    print(f"[OK] SenseVoiceエンジン推論テスト成功 (結果: '{res}')")


def test_whisper_engine():
    print("Whisperエンジンのテスト実行中 (small)...")
    engine = TranscriptionEngine(model_size="small")
    dummy_audio = np.zeros(16000, dtype=np.float32)
    res = engine.transcribe(dummy_audio, language="ja")
    assert isinstance(res, str)
    print(f"[OK] Whisperエンジン推論テスト成功 (結果: '{res}')")


if __name__ == "__main__":
    test_parsers()
    test_sensevoice_engine()
    test_whisper_engine()
    print("すべての文字起こしエンジンテストが正常に通過しました！")
