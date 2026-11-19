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
from src.runtime import (
    TRANSCRIPTION_MAX_THREADS,
    is_meaningful_transcription,
    recommend_cpu_threads,
)
from src.sensevoice_engine import SenseVoiceEngine, normalize_language


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
    with patch("src.runtime.os.cpu_count", return_value=1):
        assert recommend_cpu_threads() == 1
    with patch("src.runtime.os.cpu_count", return_value=2):
        assert recommend_cpu_threads() == 1

    # 4コア以上は論理コア数の半分（上限キャップ付き）
    with patch("src.runtime.os.cpu_count", return_value=4):
        assert recommend_cpu_threads() == 2
    with patch("src.runtime.os.cpu_count", return_value=8):
        assert recommend_cpu_threads() == 4
    with patch("src.runtime.os.cpu_count", return_value=16):
        assert recommend_cpu_threads() == TRANSCRIPTION_MAX_THREADS

    # 上限は max_threads で頭打ちになる
    with patch("src.runtime.os.cpu_count", return_value=64):
        assert recommend_cpu_threads() == TRANSCRIPTION_MAX_THREADS
        assert recommend_cpu_threads(max_threads=4) == 4

    # 上限が2未満でも頭打ちを守る（1以下は1スレッド）
    with patch("src.runtime.os.cpu_count", return_value=16):
        assert recommend_cpu_threads(max_threads=2) == 2
        assert recommend_cpu_threads(max_threads=1) == 1
        assert recommend_cpu_threads(max_threads=0) == 1
        assert recommend_cpu_threads(max_threads=-3) == 1

    # cpu_count が取得できない環境でも最低限動く
    with patch("src.runtime.os.cpu_count", return_value=None):
        assert recommend_cpu_threads() == 1

    # 実環境でも妥当な範囲に収まる
    assert 1 <= recommend_cpu_threads() <= TRANSCRIPTION_MAX_THREADS
    print(f"[OK] 推論スレッド数の自動決定テスト成功 (このPC: {recommend_cpu_threads()})")


def test_cpu_threads_propagation():
    """TranscriptionEngine が解決した cpu_threads を SenseVoice / Whisper へ渡すこと"""
    from unittest.mock import call, patch

    captured = {}

    class StubSenseVoiceEngine:
        def __init__(self, num_threads=None):
            captured["num_threads"] = num_threads

    with patch("src.transcriber.SenseVoiceEngine", StubSenseVoiceEngine), patch(
        "src.transcriber.recommend_cpu_threads", return_value=5
    ) as recommend:
        # 明示指定した値はそのまま SenseVoice へ渡る
        engine = TranscriptionEngine(model_size="sensevoice", cpu_threads=3)
        assert engine.cpu_threads == 3
        assert captured["num_threads"] == 3

        # 未指定・不正値は recommend_cpu_threads の解決結果を使う
        engine = TranscriptionEngine(model_size="sensevoice", cpu_threads=None)
        assert engine.cpu_threads == 5
        assert captured["num_threads"] == 5

        engine = TranscriptionEngine(model_size="sensevoice", cpu_threads=0)
        assert engine.cpu_threads == 5
        assert captured["num_threads"] == 5

        # 文字起こし用の上限が src.runtime の定数から渡る
        assert recommend.call_args_list == [
            call(max_threads=TRANSCRIPTION_MAX_THREADS),
            call(max_threads=TRANSCRIPTION_MAX_THREADS),
        ]

    # Whisper 経路にも同じ解決値が渡る
    with patch("src.transcriber.WhisperModel") as whisper_cls, patch(
        "src.transcriber.detect_acceleration", return_value=("cpu", "int8")
    ), patch("src.transcriber.recommend_cpu_threads", return_value=5):
        engine = TranscriptionEngine(model_size="small", device="cpu", cpu_threads=3)
        assert engine.cpu_threads == 3
        assert whisper_cls.call_args.kwargs["cpu_threads"] == 3

        engine = TranscriptionEngine(model_size="small", device="cpu", cpu_threads=None)
        assert engine.cpu_threads == 5
        assert whisper_cls.call_args.kwargs["cpu_threads"] == 5
    print("[OK] cpu_threads の伝播テスト成功")


def test_sensevoice_thread_cap():
    """SenseVoice のスレッド上限が src.runtime の定数から渡されること"""
    from unittest.mock import patch

    with patch(
        "src.sensevoice_engine.recommend_cpu_threads", return_value=3
    ) as recommend, patch.object(SenseVoiceEngine, "_load_model"):
        engine = SenseVoiceEngine()
    recommend.assert_called_once_with(max_threads=TRANSCRIPTION_MAX_THREADS)
    assert engine.num_threads == 3
    print("[OK] SenseVoice スレッド上限のテスト成功")


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


def test_whisper_transcribe_integration():
    """transcribe() のセグメント除外→結合→空文字化の統合経路を実モデルなしで検証する"""
    from types import SimpleNamespace
    from unittest.mock import patch

    audio = np.zeros(1600, dtype=np.float32)

    def segment(text="", no_speech_prob=0.05, compression_ratio=1.2, avg_logprob=-0.2):
        return SimpleNamespace(
            text=text,
            no_speech_prob=no_speech_prob,
            compression_ratio=compression_ratio,
            avg_logprob=avg_logprob,
        )

    def run(
        segments,
        language="ja",
        device="cpu",
        audio_data=None,
        detected_language="ja",
    ):
        """WhisperModel をスタブに差し替え、transcribe() を1回実行する"""
        captured = {}

        class StubWhisperModel:
            def __init__(self, model_size, **kwargs):
                captured["model_size"] = model_size
                captured["init_kwargs"] = kwargs

            def transcribe(self, waveform, **kwargs):
                captured["audio"] = waveform
                captured["transcribe_kwargs"] = kwargs
                return iter(segments), SimpleNamespace(language=detected_language)

        acceleration = ("cuda", "float16") if device == "cuda" else ("cpu", "int8")
        with patch("src.transcriber.WhisperModel", StubWhisperModel), patch(
            "src.transcriber.detect_acceleration", return_value=acceleration
        ):
            engine = TranscriptionEngine(model_size="small", device=device)
            text = engine.transcribe(
                audio if audio_data is None else audio_data, language=language
            )
        return text, engine, captured

    # 全セグメントが除外されたら空文字（幻覚の本文がそのまま返らない）
    text, engine, _ = run(
        [
            segment("ご視聴ありがとうございました", no_speech_prob=0.9),
            segment("   "),
            segment("ああああああ", compression_ratio=3.8),
            segment("ん", no_speech_prob=0.5, avg_logprob=-1.5),
        ]
    )
    assert text == ""
    assert engine.last_detected_language == "ja"

    # 信頼できるセグメントだけを strip して半角スペースで結合する
    text, engine, _ = run(
        [
            segment(" こんにちは "),
            segment("ご視聴ありがとうございました", no_speech_prob=0.95),
            segment("世界"),
        ]
    )
    assert text == "こんにちは 世界"
    assert engine.last_detected_language == "ja"

    # セグメント単位では通っても、結合結果が記号だけなら空文字にする
    text, _, _ = run([segment("。"), segment("...")])
    assert text == ""

    # しきい値の境界（0.6 は除外 / 2.4 は残す / -1.0 は残す）
    text, _, _ = run(
        [
            segment("境界0", no_speech_prob=0.6),
            segment("境界1", compression_ratio=2.4),
            segment("境界2", compression_ratio=2.41),
            segment("境界3", avg_logprob=-1.0, no_speech_prob=0.5),
            segment("境界4", avg_logprob=-1.01, no_speech_prob=0.5),
            segment("境界5", avg_logprob=-1.01, no_speech_prob=0.35),
        ]
    )
    assert text == "境界1 境界3 境界5"

    # 属性が欠けたセグメントは既定値で判定する（text が無い/None は除外）
    text, _, _ = run(
        [
            SimpleNamespace(text="属性あり"),
            SimpleNamespace(text=None),
            SimpleNamespace(),
        ]
    )
    assert text == "属性あり"

    # セグメントが1件も無い場合も空文字
    text, _, _ = run([])
    assert text == ""

    # CPU ではビームサーチを広げず、タイムスタンプも要求しない
    text, _, captured = run([segment("はい")])
    assert text == "はい"
    assert captured["model_size"] == "small"
    assert captured["audio"] is audio
    assert captured["init_kwargs"]["device"] == "cpu"
    assert captured["init_kwargs"]["compute_type"] == "int8"
    assert isinstance(captured["init_kwargs"]["cpu_threads"], int)
    kwargs = captured["transcribe_kwargs"]
    assert kwargs["language"] == "ja"
    assert kwargs["beam_size"] == 1
    assert kwargs["without_timestamps"] is True
    assert kwargs["vad_filter"] is False
    assert kwargs["condition_on_previous_text"] is False
    assert kwargs["temperature"] == 0.0

    # GPU では精度優先でビームサーチを3に広げる
    _, _, captured = run([segment("hello")], language="en", device="cuda")
    assert captured["init_kwargs"]["compute_type"] == "float16"
    assert captured["transcribe_kwargs"]["beam_size"] == 3
    assert captured["transcribe_kwargs"]["language"] == "en"

    # 空音声はモデルを呼ばずに即返す
    text, _, captured = run(
        [segment("呼ばれない")], audio_data=np.array([], dtype=np.float32)
    )
    assert text == ""
    assert "transcribe_kwargs" not in captured

    # 言語が取得できない場合は直前の検出結果を保持する
    _, engine, _ = run([segment("hello")], detected_language=None)
    assert engine.last_detected_language is None
    print("[OK] Whisper transcribe 統合経路のテスト成功")


if __name__ == "__main__":
    test_parsers()
    test_recommend_cpu_threads()
    test_cpu_threads_propagation()
    test_sensevoice_thread_cap()
    test_is_meaningful_transcription()
    test_whisper_segment_filter()
    test_whisper_transcribe_integration()
    test_detect_acceleration()
    test_sensevoice_engine()
    test_whisper_engine()
    print("すべての文字起こしエンジンテストが正常に通過しました！")
