"""音声前処理モジュールの単体テストスクリプト"""

import numpy as np
from src.processor import preprocess_audio, compute_effective_threshold


def test_preprocess_audio():
    print("音声前処理（音量ノーマライズ・DCオフセット除去）のテスト実行中...")

    # 1. 小さな音量のブーストテスト
    small_signal = np.sin(np.linspace(0, 10, 16000), dtype=np.float32) * 0.1  # ピーク0.1
    normalized = preprocess_audio(small_signal, target_peak=0.9, max_gain=6.0)
    new_peak = np.max(np.abs(normalized))
    print(f"[OK] 音量ブースト: 元ピーク 0.1 -> 正規化後ピーク {new_peak:.2f}")
    assert 0.5 <= new_peak <= 0.95

    # 2. DCオフセット（バイアス）の除去テスト
    biased_signal = small_signal + 0.3  # +0.3の直流バイアス
    corrected = preprocess_audio(biased_signal)
    mean_val = np.mean(corrected)
    print(f"[OK] DCオフセット補正: 平均値 {mean_val:.6f}")
    assert abs(mean_val) < 1e-4

    # 3. 無音（全ゼロ）の安全テスト
    zero_signal = np.zeros(16000, dtype=np.float32)
    zero_out = preprocess_audio(zero_signal)
    assert np.all(zero_out == 0)
    print("[OK] 無音信号の安全処理確認")


def test_effective_threshold_adaptation():
    """騒がしい環境でも閾値が追従して上がること"""
    # 静かな環境: 下限 0.003
    assert compute_effective_threshold(0.006, 0.001) == 0.003
    # 騒がしい環境: 基準値の3倍 (0.018) まで許容
    assert abs(compute_effective_threshold(0.006, 0.010) - 0.018) < 1e-9
    # 極端に大きいノイズフロアでも上限を超えない
    assert abs(compute_effective_threshold(0.006, 1.0) - 0.018) < 1e-9
    print("[OK] VAD有効閾値の環境適応テスト成功")


def test_resample_audio():
    """16kHz非対応マイク向けのリサンプリング"""
    from src.audio import resample_audio

    # 16kHz はそのまま
    audio = np.full(1600, 0.5, dtype=np.float32)
    same = resample_audio(audio, 16000, 16000)
    assert len(same) == 1600
    assert np.allclose(same, 0.5)

    # 48kHz -> 16kHz で長さが 1/3 になる
    signal_48k = (np.sin(np.linspace(0, 2 * np.pi * 440, 4800)) * 0.5).astype(np.float32)
    out = resample_audio(signal_48k, 48000, 16000)
    assert out.dtype == np.float32
    assert len(out) == 1600
    assert np.max(np.abs(out)) <= 0.6

    # 44.1kHz -> 16kHz（非整数倍）
    out2 = resample_audio(signal_48k[:4410], 44100, 16000)
    assert len(out2) == 1600

    # 空データは空で返る
    assert len(resample_audio(np.zeros(0, dtype=np.float32), 48000, 16000)) == 0
    print("[OK] リサンプリングのテスト成功")


def test_input_sample_rate_fallback():
    """16kHzを受け付けないマイクで既定レートへフォールバックすること"""
    from src import audio as audio_module

    recorder = audio_module.AudioRecorder()
    original_check = audio_module.sd.check_input_settings
    original_query = audio_module.sd.query_devices

    try:
        # 16kHz が使える場合はそのまま
        audio_module.sd.check_input_settings = lambda **kwargs: None
        assert recorder._select_input_sample_rate(None) == 16000

        # 16kHz が使えず、デバイス既定が 48kHz の場合
        def raise_invalid_rate(**kwargs):
            raise RuntimeError("Invalid sample rate")

        audio_module.sd.check_input_settings = raise_invalid_rate
        audio_module.sd.query_devices = lambda *args, **kwargs: {"default_samplerate": 48000.0}
        assert recorder._select_input_sample_rate(None) == 48000
        assert recorder.sample_rate == 16000  # 内部処理は16kHzのまま

        # 既定レートも取得できない場合は元のエラーを伝える
        def broken_query(*args, **kwargs):
            raise RuntimeError("no device")

        audio_module.sd.query_devices = broken_query
        try:
            recorder._select_input_sample_rate(None)
            raise AssertionError("例外が送出されるべきです")
        except RuntimeError as e:
            assert "sample rate" in str(e)

        print("[OK] マイクのサンプリングレートフォールバックテスト成功")
    finally:
        audio_module.sd.check_input_settings = original_check
        audio_module.sd.query_devices = original_query


def test_audio_queue_is_bounded():
    """キュー上限を超えた場合に古いチャンクが破棄され、時刻が付与されること"""
    from src.audio import AudioRecorder

    recorder = AudioRecorder(sample_rate=16000)
    recorder._is_recording = True
    for _ in range(AudioRecorder.MAX_QUEUED_CHUNKS + 10):
        recorder._callback(np.zeros(16, dtype=np.float32), 16, None, None)

    assert recorder.audio_queue.qsize() == AudioRecorder.MAX_QUEUED_CHUNKS
    assert recorder.dropped_chunks == 10

    timestamp, chunk = recorder.get_chunk_with_timestamp(timeout=0.1)
    assert timestamp is not None
    assert chunk is not None and len(chunk) == 16
    print("[OK] 音声キュー上限・タイムスタンプ付与テスト成功")


def test_audio_processor_diarization():
    print("AudioProcessor と SpeakerDiarizer 連携テスト実行中...")
    from unittest.mock import MagicMock
    from src.processor import AudioProcessor
    from src.audio import AudioRecorder
    from src.diarizer import SpeakerDiarizer

    recorder = AudioRecorder(sample_rate=16000)
    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = "こんにちは、テストです。"

    results = []
    def callback(text: str):
        results.append(text)

    diarizer = SpeakerDiarizer(threshold=0.55)
    diarizer.identify_speaker = MagicMock(return_value="話者1")

    processor = AudioProcessor(
        recorder=recorder,
        transcriber=mock_transcriber,
        on_transcription_callback=callback,
        diarizer=diarizer,
    )
    processor.enable_diarization = True

    dummy_speech = [np.ones(16000, dtype=np.float32) * 0.1]
    processor._process_speech(dummy_speech, sample_rate=16000)

    assert len(results) == 1
    assert results[0] == "[話者1] こんにちは、テストです。"
    print(f"[OK] 話者タグ付与連携テスト成功: {results[0]}")


def test_audio_processor_vad_and_short_speech():
    print("AudioProcessor VAD閾値適正化および短文処理テスト実行中...")
    from unittest.mock import MagicMock
    from src.processor import AudioProcessor
    from src.audio import AudioRecorder

    recorder = AudioRecorder(sample_rate=16000)
    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = "はい"

    results = []
    def callback(text: str):
        results.append(text)

    activity_log = []
    def on_activity(rms: float, is_active: bool):
        activity_log.append((rms, is_active))

    processor = AudioProcessor(
        recorder=recorder,
        transcriber=mock_transcriber,
        on_transcription_callback=callback,
        silence_threshold=0.006,
        min_speech_duration_sec=0.3,
        on_activity_callback=on_activity,
    )

    # 0.4秒分の音声（旧仕様1.0sでは切り捨てられていた短い相槌）
    short_speech = [np.ones(int(16000 * 0.4), dtype=np.float32) * 0.008]
    processor._process_speech(short_speech, sample_rate=16000)

    assert len(results) == 1
    assert results[0] == "はい"
    print(f"[OK] 短文発話(0.4秒)の正常処理確認: '{results[0]}'")


if __name__ == "__main__":
    test_preprocess_audio()
    test_effective_threshold_adaptation()
    test_resample_audio()
    test_input_sample_rate_fallback()
    test_audio_queue_is_bounded()
    test_audio_processor_diarization()
    test_audio_processor_vad_and_short_speech()
    print("すべての音声前処理・プロセッサテストが正常に通過しました！")
