"""音声前処理モジュールの単体テストスクリプト"""

import numpy as np
from src.processor import preprocess_audio


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


if __name__ == "__main__":
    test_preprocess_audio()
    test_audio_processor_diarization()
    print("すべての音声前処理・プロセッサテストが正常に通過しました！")
