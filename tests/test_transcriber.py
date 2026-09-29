"""文字起こしエンジンのテストスクリプト"""

import numpy as np
from src.transcriber import parse_language_code, parse_model_size, TranscriptionEngine


def test_parsers():
    assert parse_language_code("自動検出 (auto)") is None
    assert parse_language_code("日本語 (ja)") == "ja"
    assert parse_language_code("英語 (en)") == "en"
    assert parse_language_code("中国語 (zh)") == "zh"

    assert parse_model_size("large-v3-turbo (最高精度・推奨)") == "large-v3-turbo"
    assert parse_model_size("small (高精度・軽量)") == "small"
    print("[OK] パース関数のテスト成功")


def test_engine_inference():
    print("tinyモデルの読み込み中...")
    engine = TranscriptionEngine(model_size="tiny")
    dummy_audio = np.zeros(16000 * 2, dtype=np.float32)  # 2秒分の無音
    res = engine.transcribe(dummy_audio)
    assert isinstance(res, str)
    print(f"[OK] 推論テスト成功 (結果: '{res}')")


if __name__ == "__main__":
    test_parsers()
    test_engine_inference()
    print("すべての文字起こしテストが正常に通過しました！")
