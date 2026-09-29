"""音源分離モジュールのテストスクリプト"""

import numpy as np
from src.separator import SpeechSeparator
from src.processor import AudioProcessor
from src.audio import AudioRecorder
from src.transcriber import TranscriptionEngine


def test_speech_separator_fallback():
    print("SpeechSeparatorのフォールバック・安全性テスト中...")
    separator = SpeechSeparator()
    # モデル未ロード時でも安全に元音声を返すか
    dummy_audio = np.zeros(16000, dtype=np.float32)
    tracks = separator.separate(dummy_audio)
    assert len(tracks) >= 1
    print("[OK] SpeechSeparatorの安全性テスト成功")


def test_processor_with_separation_flag():
    print("AudioProcessorの分離フラグ切り替えテスト中...")
    recorder = AudioRecorder()
    engine = TranscriptionEngine(model_size="sensevoice")
    separator = SpeechSeparator()

    results = []
    processor = AudioProcessor(
        recorder=recorder,
        transcriber=engine,
        on_transcription_callback=lambda text: results.append(text),
        separator=separator,
    )
    assert processor.enable_separation is False
    processor.enable_separation = True
    assert processor.enable_separation is True
    print("[OK] 音源分離フラグのトグル制御テスト成功")


if __name__ == "__main__":
    test_speech_separator_fallback()
    test_processor_with_separation_flag()
    print("すべての音源分離テストが正常に通過しました！")
