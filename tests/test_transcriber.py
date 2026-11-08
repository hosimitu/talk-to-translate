import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.transcriber import (
    detect_acceleration,
    parse_language_code,
    parse_model_size,
    TranscriptionEngine,
)
from src.sensevoice_engine import (
    is_meaningful_transcription,
    normalize_language,
    recommend_cpu_threads,
)


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


def test_detect_acceleration():
    """GPU/CPUの自動判定結果が整合していること"""
    device, compute_type = detect_acceleration()
    assert device in ("cpu", "cuda")
    assert compute_type in ("int8", "float16")
    if device == "cuda":
        assert compute_type == "float16"
    else:
        assert compute_type == "int8"
    print(f"[OK] 推論デバイス自動判定テスト成功 (device={device}, compute_type={compute_type})")


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


def test_recommend_cpu_threads():
    """CPUコア数から推論スレッド数が妥当に決まること（モデル非依存）"""
    from unittest.mock import patch

    # 2コア以下の低スペックPCでは録音・UIスレッドを圧迫しないよう1スレッド
    with patch("src.sensevoice_engine.os.cpu_count", return_value=1):
        assert recommend_cpu_threads() == 1
    with patch("src.sensevoice_engine.os.cpu_count", return_value=2):
        assert recommend_cpu_threads() == 1

    # 4コア以上は論理コア数の半分（上限キャップ付き）
    with patch("src.sensevoice_engine.os.cpu_count", return_value=4):
        assert recommend_cpu_threads() == 2
    with patch("src.sensevoice_engine.os.cpu_count", return_value=8):
        assert recommend_cpu_threads() == 4
    with patch("src.sensevoice_engine.os.cpu_count", return_value=16):
        assert recommend_cpu_threads() == 8

    # 上限は max_threads で頭打ちになる
    with patch("src.sensevoice_engine.os.cpu_count", return_value=64):
        assert recommend_cpu_threads() == 8
        assert recommend_cpu_threads(max_threads=4) == 4

    # cpu_count が取得できない環境でも最低限動く
    with patch("src.sensevoice_engine.os.cpu_count", return_value=None):
        assert recommend_cpu_threads() == 1

    # 実環境でも妥当な範囲に収まる
    assert 1 <= recommend_cpu_threads() <= 8
    print(f"[OK] 推論スレッド数の自動決定テスト成功 (このPC: {recommend_cpu_threads()})")


def test_is_meaningful_transcription():
    """句読点・記号だけの幻覚出力を弾き、実際の語は通すこと"""
    # 無音・雑音に対して返る記号だけの出力は無意味
    assert is_meaningful_transcription("。") is False
    assert is_meaningful_transcription("...") is False
    assert is_meaningful_transcription("！？") is False
    assert is_meaningful_transcription("、。") is False
    assert is_meaningful_transcription("") is False
    assert is_meaningful_transcription("   ") is False
    assert is_meaningful_transcription("♪") is False
    assert is_meaningful_transcription(None) is False

    # 短い相槌や1文字の語は通す
    assert is_meaningful_transcription("はい") is True
    assert is_meaningful_transcription("うん") is True
    assert is_meaningful_transcription("OK") is True
    assert is_meaningful_transcription("1") is True
    assert is_meaningful_transcription("你好") is True
    assert is_meaningful_transcription("안녕") is True
    assert is_meaningful_transcription("こんにちは。") is True
    print("[OK] 記号のみ出力の判定テスト成功")


def test_whisper_segment_filter():
    """Whisperのハルシネーションセグメントが除外されること"""
    from types import SimpleNamespace

    engine = TranscriptionEngine.__new__(TranscriptionEngine)

    good = SimpleNamespace(text="こんにちは", no_speech_prob=0.05, compression_ratio=1.2, avg_logprob=-0.2)
    assert engine._is_reliable_segment(good) is True

    # 空文字
    empty = SimpleNamespace(text="   ", no_speech_prob=0.05, compression_ratio=1.2, avg_logprob=-0.2)
    assert engine._is_reliable_segment(empty) is False

    # 無音確率が高い
    silent = SimpleNamespace(text="ご視聴ありがとうございました", no_speech_prob=0.9, compression_ratio=1.5, avg_logprob=-0.4)
    assert engine._is_reliable_segment(silent) is False

    # 同一フレーズの反復暴走（compression_ratio が跳ね上がる）
    looped = SimpleNamespace(text="ああああああああ", no_speech_prob=0.2, compression_ratio=3.8, avg_logprob=-0.5)
    assert engine._is_reliable_segment(looped) is False

    # 低確信度かつ無音の可能性あり
    noisy = SimpleNamespace(text="ん", no_speech_prob=0.5, compression_ratio=1.1, avg_logprob=-1.5)
    assert engine._is_reliable_segment(noisy) is False

    # avg_logprob が低くても無音確率が低ければ発話として残す
    fast_speech = SimpleNamespace(text="はい", no_speech_prob=0.1, compression_ratio=1.0, avg_logprob=-1.2)
    assert engine._is_reliable_segment(fast_speech) is True
    print("[OK] Whisperハルシネーション除外のテスト成功")


if __name__ == "__main__":
    test_parsers()
    test_recommend_cpu_threads()
    test_is_meaningful_transcription()
    test_whisper_segment_filter()
    test_detect_acceleration()
    test_sensevoice_engine()
    test_whisper_engine()
    print("すべての文字起こしエンジンテストが正常に通過しました！")
