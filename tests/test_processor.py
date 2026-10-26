"""音声前処理モジュールの単体テストスクリプト"""

import time

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


def test_compute_boost_gain():
    """ノイズフロアとの差（SNR余裕）に応じた増幅率の制限"""
    from src.processor import compute_boost_gain

    # 明瞭な音声（フロアと十分差がある）: 通常の正規化レート
    assert compute_boost_gain(0.5, 0.001) < 2.0

    # フロアと区別できない小さなピーク: ほぼ増幅しない
    assert compute_boost_gain(0.002, 0.0015) <= 1.2

    # わずかに浮いているだけの中間: 中間の上限
    gain = compute_boost_gain(0.010, 0.004)
    assert 1.2 < gain <= 2.0

    # フロアを大きく超える小音量: 音量正規化どおりに増幅する
    assert compute_boost_gain(0.02, 0.001) > 2.0

    # ほぼ無音・ゼロ: 増幅しない
    assert compute_boost_gain(0.00005, 0.001) == 1.0
    assert compute_boost_gain(0.0, 0.001) == 1.0
    print("[OK] 音量ブースト制限テスト成功")


def test_format_elapsed_time():
    """経過秒の表示形式 (MM:SS / H:MM:SS)"""
    from src.processor import format_elapsed_time

    assert format_elapsed_time(0) == "00:00"
    assert format_elapsed_time(5) == "00:05"
    assert format_elapsed_time(65) == "01:05"
    assert format_elapsed_time(600) == "10:00"
    assert format_elapsed_time(3671) == "1:01:11"
    assert format_elapsed_time(-1.5) == "00:00"
    print("[OK] 経過時間フォーマットテスト成功")


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
class _FakeRecorder:
    """チャンク (取得時刻, 音声) のリストをキューから返すだけの簡易レコーダー"""

    sample_rate = 16000
    dropped_chunks = 0

    def __init__(self, items):
        self._items = list(items)

    def get_chunk_with_timestamp(self, timeout=0.1):
        if not self._items:
            import time as _time
            _time.sleep(0.01)  # チャンク切れ時のスピン待機を軽減
            return None, None
        return self._items.pop(0)


def _run_worker_loop(processor, wait_sec=1.0):
    """_worker_loop をスレッドで実行し、全チャンク消化後に停止する"""
    import threading

    thread = threading.Thread(target=processor._worker_loop, daemon=True)
    thread.start()
    time.sleep(wait_sec)
    processor._is_running = False
    thread.join(timeout=2.0)


def _build_processor(items, mock_transcriber):
    """_worker_loop を直接走らせるためのプロセッサを構築する"""
    from src.processor import AudioProcessor

    results = []
    processor = AudioProcessor(
        recorder=_FakeRecorder(items),
        transcriber=mock_transcriber,
        on_transcription_callback=results.append,
    )
    processor._is_running = True
    return processor, results


def test_split_overlap_and_timestamps():
    """8秒分割で末尾がオーバーラップし、発話に時刻接頭辞が付くこと"""
    from unittest.mock import MagicMock
    import re as _re
    from src.processor import AudioProcessor

    sr = 16000
    silence = np.zeros(16000, dtype=np.float32)
    speech = (np.sin(np.linspace(0, 2 * np.pi * 440, 16000)) * 0.1).astype(np.float32)

    items = []
    for t in range(2):                       # 無音 2秒 (t=0, 1)
        items.append((float(t), silence.copy()))
    for t in range(2, 22):                   # 発話 20秒 (t=2〜21)
        items.append((float(t), speech.copy()))
    for t in range(22, 25):                  # 途中無音 (発話終了判定)
        items.append((float(t), silence.copy()))

    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = "こんにちは"
    segments = []
    original = mock_transcriber.transcribe
    mock_transcriber.transcribe = lambda audio, language=None: (
        segments.append(audio.copy()) or original(audio, language=language)
    )

    results = []
    processor = AudioProcessor(
        recorder=_FakeRecorder(items),
        transcriber=mock_transcriber,
        on_transcription_callback=results.append,
    )
    processor._is_running = True
    processor._recording_started_at = 0.0  # start() 相当
    _run_worker_loop(processor)

    # 20秒の発話 → 最低2回に分割される
    assert len(segments) >= 2
    # 先頭区間: 発話前に構えた pre_buffer 1チャンク（無音 t=1）+ 発話 t=2〜9 の 9チャンク
    assert len(segments[0]) == int(9.0 * sr)
    # 以降の区間も最大発話時間＋オーバーラップ分を超えない
    for seg in segments[1:]:
        assert len(seg) <= int((8.0 + 0.5) * sr)
    # 分割部は削除ではなく引き継ぎのため、区間のつながりで音声が欠けていない
    # （2区間目の先頭は前区間末尾の直後のチャンク＝発話波形そのもの。ブースト×6済み）
    assert np.allclose(segments[1][:sr], speech * 6.0, atol=1e-5)
    # 時刻接頭辞: 最初の発話は t=2.0 から → "[00:02] " で始まる
    assert results[0].startswith("[00:02] こんにちは")
    assert all(_re.match(r"^\[\d{1,3}:\d{2}(?::\d{2})?\] ", text) for text in results)
    print(f"[OK] 8秒分割オーバーラップとタイムスタンプのテスト成功: {len(segments)}区間")


def test_pre_buffer_seconds_based():
    """発話前バッファが「秒」基準で保持されること（チャンク長がデバイス依存でも）

    `0.5秒チャンクのデバイスでは、旧実装（固定3チャンク）では1.5秒もさかのぼる一方、
    チャンク長が長い場合は語頭欠け防止に必要なのは pre_buffer_sec(0.3秒) 分。
    0.3秒分が保持されるなら 1 チャンク丸ごと（0.5秒）が残る。
    """
    from unittest.mock import MagicMock

    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = "こんにちは"
    sr = 16000
    chunk_len = int(sr * 0.5)
    silence = np.zeros(chunk_len, dtype=np.float32)
    speech = (np.sin(np.linspace(0, 2 * np.pi * 440, chunk_len)) * 0.1).astype(np.float32)

    items = []
    for i in range(4):                       # 無音 4チャンク (t=0〜1.5)
        items.append((i * 0.5, silence.copy()))
    items.append((2.0, speech.copy()))       # 発話 (t=2.0)
    items.append((2.5, speech.copy()))       # 発話 (t=2.5)
    items.append((3.0, silence.copy()))      # 発話終了判定開始
    items.append((4.0, silence.copy()))      # 1秒無音で確定

    processor, results = _build_processor(items, mock_transcriber)
    _run_worker_loop(processor)

    assert len(processor.transcriber.transcribe.call_args_list) == 1
    audio = processor.transcriber.transcribe.call_args_list[0][0][0]
    # 保持チャンク: pre_buffer 0.3秒→1チャンク + 発話2 + 終了判定用無音2 = 5チャンク (2.5秒)
    assert len(audio) == chunk_len * 5
    # 冒頭に発話前 0.3 秒以上（実装では1チャンク=0.5秒）が含まれている
    assert len(results) == 1
    print(f"[OK] 秒基準の発話前バッファのテスト成功 (長さ {len(audio)} サンプル = {len(audio)/sr:.1f}秒)")


if __name__ == "__main__":
    test_preprocess_audio()
    test_effective_threshold_adaptation()
    test_compute_boost_gain()
    test_format_elapsed_time()
    test_resample_audio()
    test_input_sample_rate_fallback()
    test_audio_queue_is_bounded()
    test_split_overlap_and_timestamps()
    test_pre_buffer_seconds_based()
    test_audio_processor_diarization()
    test_audio_processor_vad_and_short_speech()
    print("すべての音声前処理・プロセッサテストが正常に通過しました！")
