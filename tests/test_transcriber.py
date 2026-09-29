"""文字起こしエンジンのテストスクリプト"""

import numpy as np
from src.transcriber import parse_language_code, parse_model_size, TranscriptionEngine


def test_parsers():
    assert parse_language_code("自動検出 (auto)") is None
    assert parse_language_code("日本語 (ja)") == "ja"
    assert parse_language_code("英語 (en)") == "en"
    assert parse_language_code("中国語 (zh)") == "zh"

    assert parse_model_size("SenseVoice-Small (超高速・日中英推奨)") == "sensevoice"
    assert parse_model_size("Whisper: large-v3-turbo (高精度)") == "large-v3-turbo"
    assert parse_model_size("Whisper: small (軽量)") == "small"
    print("[OK] パース関数のテスト成功")


def test_sensevoice_engine():
    print("SenseVoiceエンジンのテスト実行中...")
    engine = TranscriptionEngine(model_size="sensevoice")
    dummy_audio = np.zeros(16000, dtype=np.float32)
    res = engine.transcribe(dummy_audio, language="ja")
    assert isinstance(res, str)
    print(f"[OK] SenseVoiceエンジン推論テスト成功 (結果: '{res}')")


if __name__ == "__main__":
    test_parsers()
    test_sensevoice_engine()
    print("すべての文字起こしエンジンテストが正常に通過しました！")
