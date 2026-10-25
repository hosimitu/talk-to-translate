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
    test_audio_queue_is_bounded()
    test_audio_processor_diarization()
    test_audio_processor_vad_and_short_speech()
    print("すべての音声前処理・プロセッサテストが正常に通過しました！")
