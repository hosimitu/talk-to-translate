"""音声前処理モジュールの単体テストスクリプト"""

import threading
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


def test_voiced_threshold_boundary():
    """RMS が有効閾値と同値なら無音扱い（閾値を上回る場合だけ有音）"""
    from src.processor import compute_effective_threshold, is_voiced_chunk

    threshold = compute_effective_threshold(0.006, 0.002)
    assert abs(threshold - 0.005) < 1e-12
    assert is_voiced_chunk(threshold, threshold) is False
    assert is_voiced_chunk(threshold + 1e-6, threshold) is True
    assert is_voiced_chunk(threshold - 1e-6, threshold) is False
    print("[OK] 有音判定の閾値境界テスト成功")


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
        # 全チャンクを返し終えたことをテスト側へ通知する（固定 sleep を避ける）
        self.exhausted = threading.Event()

    def get_chunk_with_timestamp(self, timeout=0.1):
        if not self._items:
            self.exhausted.set()
            time.sleep(0.005)  # 停止指示を受け取るまでのビジーループを軽減
            return None, None
        return self._items.pop(0)


def _run_worker_loop(processor, timeout=5.0):
    """_worker_loop をスレッドで実行し、全チャンク消化後に決定的に停止する"""
    thread = threading.Thread(target=processor._worker_loop, daemon=True)
    thread.start()
    assert processor.recorder.exhausted.wait(timeout), (
        "ワーカーが音声チャンクを消化しませんでした"
    )
    processor._is_running = False
    thread.join(timeout=2.0)
    assert not thread.is_alive(), "ワーカースレッドが停止しませんでした"


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
    """8秒分割で末尾が引き継がれ、発話に時刻接頭辞が付くこと"""
    from unittest.mock import MagicMock
    import re as _re
    from src.processor import AudioProcessor, compute_boost_gain, format_elapsed_time

    sr = 16000
    lead_silence_chunks = 2
    speech_chunks = 20
    trailing_silence_chunks = 3

    silence = np.zeros(sr, dtype=np.float32)
    speech = (np.sin(np.linspace(0, 2 * np.pi * 440, sr)) * 0.1).astype(np.float32)
    chunk_sec = len(speech) / sr

    items = []
    for i in range(lead_silence_chunks):  # 発話前の無音
        items.append((float(i), silence.copy()))
    for i in range(lead_silence_chunks, lead_silence_chunks + speech_chunks):  # 発話
        items.append((float(i), speech.copy()))
    for i in range(  # 発話終了判定用の無音
        lead_silence_chunks + speech_chunks,
        lead_silence_chunks + speech_chunks + trailing_silence_chunks,
    ):
        items.append((float(i), silence.copy()))

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

    # どのチャンクがどの区間に渡ったかを記録し、被覆（ギャップ・overlap）を検証する
    chunk_indices = {id(chunk): i for i, (_, chunk) in enumerate(items)}
    spans = []
    original_process = processor._process_speech

    def record_process(buffer, sample_rate, speech_start_time=None):
        indices = [chunk_indices[id(chunk)] for chunk in buffer]
        spans.append((indices[0], indices[-1]))
        return original_process(buffer, sample_rate, speech_start_time)

    processor._process_speech = record_process
    initial_noise_floor = processor.noise_floor
    _run_worker_loop(processor)

    pre_chunks = max(1, int(round(processor.pre_buffer_sec / chunk_sec)))
    overlap_chunks = int(np.ceil(processor.split_overlap_sec / chunk_sec))
    speech_start_index = lead_silence_chunks
    speech_end_index = lead_silence_chunks + speech_chunks - 1

    # 20秒の発話 → 最低2回に分割される
    assert len(segments) >= 2
    assert len(spans) == len(segments)

    # 先頭区間は pre_buffer 分だけ発話開始前にさかのぼる
    assert spans[0][0] == speech_start_index - pre_chunks
    # 区間はギャップなくつながり、重なりも設定値＋チャンク粒度以内に収まる
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
        assert next_start <= prev_end + 1, "区間の間にギャップがあります"
        assert next_start >= prev_end + 1 - overlap_chunks, (
            "オーバーラップが上限を超えています"
        )
    # 末尾の発話まで途切れずに含まれる
    assert spans[-1][1] >= speech_end_index, "末尾の発話が含まれていません"

    # 先頭区間は最大発話時間に達して分割され、チャンク1個分の余裕に収まる
    assert len(segments[0]) >= int(processor.max_speech_duration_sec * sr)
    assert len(segments[0]) <= int(
        (processor.max_speech_duration_sec + chunk_sec) * sr
    )
    # 以降の区間も最大発話時間＋オーバーラップ＋チャンク粒度を超えない
    for seg in segments[1:]:
        assert len(seg) <= int(
            (
                processor.max_speech_duration_sec
                + processor.split_overlap_sec
                + chunk_sec
            )
            * sr
        )

    # 引き継ぎ区間の先頭は前区間末尾のチャンクそのもの（ブースト率は実装から導出）
    expected_gain = compute_boost_gain(
        float(np.max(np.abs(speech - np.mean(speech)))), initial_noise_floor
    )
    assert np.allclose(segments[1][:sr], speech * expected_gain, atol=1e-5)

    # 時刻接頭辞: 最初の発話の時刻から始まる
    expected_prefix = f"[{format_elapsed_time(float(speech_start_index))}] "
    assert results[0].startswith(expected_prefix + "こんにちは")
    assert all(_re.match(r"^\[\d{1,3}:\d{2}(?::\d{2})?\] ", text) for text in results)
    print(f"[OK] 8秒分割オーバーラップとタイムスタンプのテスト成功: {len(segments)}区間")


def test_transient_noise_is_rejected():
    """単発ノイズ（実発話が最小発話時間未満）では推論を呼ばず破棄すること

    発話前バッファ(0.3秒)と末尾の無音待機(0.8秒)により区間長は1.1秒を超えてしまう。
    実際に有音だったのは1チャンク(26ms)だけなので、区間長ではなく有音時間で
    判定しないと、打鍵音・咳払いのたびに推論が走ってしまう。
    """
    from unittest.mock import MagicMock

    chunk_len = 416  # 実測した PortAudio の1チャンク（約26ms）
    chunk_sec = chunk_len / 16000
    silence = np.zeros(chunk_len, dtype=np.float32)
    click = (np.sin(np.linspace(0, 2 * np.pi * 1000, chunk_len)) * 0.5).astype(np.float32)

    items = []
    # 事前の無音（pre_buffer 0.3秒 以上）
    for i in range(12):
        items.append((i * chunk_sec, silence.copy()))
    # 打鍵音などの単発ノイズ1チャンクのみ
    items.append((12 * chunk_sec, click.copy()))
    # 末尾の無音待機（silence_duration_sec = 0.8秒 を超えるまで）
    for i in range(13, 60):
        items.append((i * chunk_sec, silence.copy()))

    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = "幻の文字起こし"
    processor, results = _build_processor(items, mock_transcriber)
    _run_worker_loop(processor)

    # 推論が一度も呼ばれないこと（=CPU を使わない）
    assert mock_transcriber.transcribe.call_count == 0, (
        f"単発ノイズで推論が走りました: {mock_transcriber.transcribe.call_count} 回"
    )
    assert results == []
    print("[OK] 単発ノイズの棄却（推論呼び出し0回）テスト成功")


def test_short_acknowledgement_is_kept():
    """短い相槌（0.4秒）は有音時間が閾値以上なので認識されること"""
    from unittest.mock import MagicMock

    chunk_len = 416
    chunk_sec = chunk_len / 16000
    silence = np.zeros(chunk_len, dtype=np.float32)
    speech = (np.sin(np.linspace(0, 2 * np.pi * 300, chunk_len)) * 0.1).astype(np.float32)

    items = []
    for i in range(12):
        items.append((i * chunk_sec, silence.copy()))
    # 0.4秒分（約15チャンク）の有音
    for i in range(12, 27):
        items.append((i * chunk_sec, speech.copy()))
    for i in range(27, 60):
        items.append((i * chunk_sec, silence.copy()))

    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = "はい"
    processor, results = _build_processor(items, mock_transcriber)
    _run_worker_loop(processor)

    assert mock_transcriber.transcribe.call_count == 1
    assert len(results) == 1 and "はい" in results[0]
    print(f"[OK] 短い相槌(0.4秒)の認識継続テスト成功: '{results[0]}'")


def test_finalize_segment_min_duration_boundary():
    """最小発話時間ちょうどは通過し、1サンプルでも下回れば棄却されること"""
    from unittest.mock import MagicMock
    from src.processor import AudioProcessor
    from src.audio import AudioRecorder

    sr = 16000
    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = "はい"
    results = []
    processor = AudioProcessor(
        recorder=AudioRecorder(sample_rate=sr),
        transcriber=mock_transcriber,
        on_transcription_callback=results.append,
        min_speech_duration_sec=0.3,
    )
    buffer = [np.full(1600, 0.1, dtype=np.float32)]

    # 0.3秒ちょうどは通過する
    processor._finalize_segment(buffer, sr, 0.0, voiced_samples=int(0.3 * sr))
    assert mock_transcriber.transcribe.call_count == 1
    assert results == ["はい"]

    # 1サンプル下回れば棄却する
    processor._finalize_segment(buffer, sr, 0.0, voiced_samples=int(0.3 * sr) - 1)
    assert mock_transcriber.transcribe.call_count == 1
    assert results == ["はい"]
    print("[OK] 最小発話時間の境界テスト成功")


def test_hard_split_tail_ignores_silence():
    """8秒分割の引き継ぎで、末尾の無音を有音時間に数えないこと

    分割直後の区間は有音0.1秒しかない。引き継ぎ区間長0.4秒を有音扱いすると
    最小発話時間を誤って満たしてしまう。
    """
    from unittest.mock import MagicMock

    sr = 16000
    chunk_len = int(sr * 0.1)
    speech = (np.sin(np.linspace(0, 2 * np.pi * 440, chunk_len)) * 0.1).astype(np.float32)
    silence = np.zeros(chunk_len, dtype=np.float32)

    items = []
    # 0.0〜7.7秒は発話。8.0秒到達でハードスプリットする
    # （本番同様に 0 以外の時刻から開始する。speech_start_time=0.0 は分割判定で偽扱いになるため）
    base = 100.0
    for i in range(78):
        items.append((base + i * 0.1, speech.copy()))
    # 7.8秒以降は無音（分割時に引き継ぐ末尾0.4秒へ無音が混ざる）
    for i in range(78, 90):
        items.append((base + i * 0.1, silence.copy()))

    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = "こんにちは"
    processor, results = _build_processor(items, mock_transcriber)
    _run_worker_loop(processor)

    # 先頭区間だけが推論され、有音0.1秒の引き継ぎ区間は棄却される
    assert mock_transcriber.transcribe.call_count == 1
    assert len(results) == 1
    print("[OK] 8秒分割の引き継ぎで無音を有音計上しないテスト成功")


def test_speech_dropping_below_threshold_is_rejected():
    """途中で閾値を下回った小振幅発話は、有音時間が最小未満なら棄却されること"""
    from unittest.mock import MagicMock

    sr = 16000
    chunk_len = int(sr * 0.1)
    speech = (np.sin(np.linspace(0, 2 * np.pi * 440, chunk_len)) * 0.1).astype(np.float32)
    quiet = np.full(chunk_len, 0.005, dtype=np.float32)  # 有効閾値(0.00775)未満
    silence = np.zeros(chunk_len, dtype=np.float32)

    items = []
    # 0.2秒だけ閾値を超える
    for i in range(2):
        items.append((i * 0.1, speech.copy()))
    # 以降は閾値付近の小振幅（有音扱いされない）
    for i in range(2, 11):
        items.append((i * 0.1, quiet.copy()))
    for i in range(11, 20):
        items.append((i * 0.1, silence.copy()))

    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = "幻の文字起こし"
    processor, results = _build_processor(items, mock_transcriber)
    _run_worker_loop(processor)

    assert mock_transcriber.transcribe.call_count == 0
    assert results == []
    print("[OK] 閾値付近で落ちた小振幅発話の棄却テスト成功")


def test_soft_split_on_short_pause():
    """一定時間話した後の短い息継ぎで区切られ、8秒を待たないこと

    無音 0.5 秒は silence_duration_sec(0.8秒) 未満なので、ソフトスプリットが
    無ければ区間は確定しない。soft_split_duration_sec(5.0秒) を超えて話した後の
    息継ぎでは確定することを確かめる。
    """
    from unittest.mock import MagicMock

    sr = 16000
    silence = np.zeros(sr, dtype=np.float32)
    speech = (np.sin(np.linspace(0, 2 * np.pi * 440, sr)) * 0.1).astype(np.float32)

    items = []
    # 発話 6秒 (t=0〜5) — soft_split_duration_sec(5.0秒) を超える
    for t in range(6):
        items.append((float(t), speech.copy()))
    # 0.25秒の息継ぎを3つ (t=6.0, 6.25, 6.5) — 合計0.5秒は0.8秒未満
    quarter = sr // 4
    for i, t in enumerate((6.0, 6.25, 6.5)):
        items.append((t, np.zeros(quarter, dtype=np.float32)))
    # 以降は無音を続ける
    for t in (7.0, 8.0, 9.0):
        items.append((t, silence.copy()))

    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = "こんにちは"
    segments = []
    original = mock_transcriber.transcribe

    def record(audio, language=None):
        segments.append(audio.copy())
        return original(audio, language=language)

    mock_transcriber.transcribe = record

    processor, results = _build_processor(items, mock_transcriber)
    processor._recording_started_at = 0.0
    _run_worker_loop(processor)

    # 8秒のハードスプリットを待たず、息継ぎの時点で区間が確定している
    assert len(segments) >= 1, "ソフトスプリットで区間が確定しませんでした"
    assert len(results) >= 1
    # 最初の区間はハードスプリットの最大発話時間より明確に短い
    assert len(segments[0]) < int(processor.max_speech_duration_sec * sr)
    print(
        f"[OK] 息継ぎによるソフトスプリットのテスト成功: "
        f"{len(segments)}区間 / 先頭 {len(segments[0])/sr:.2f}秒"
    )


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
    # 保持チャンク数は pre_buffer_sec とシナリオ（発話2＋終了判定用無音2）から導出する
    chunk_sec = chunk_len / sr
    pre_chunks = max(1, int(round(processor.pre_buffer_sec / chunk_sec)))
    speech_chunks = 2
    trailing_silence_chunks = 2
    expected_chunks = pre_chunks + speech_chunks + trailing_silence_chunks
    assert len(audio) == chunk_len * expected_chunks
    # 冒頭の pre_buffer 分は無音（DC除去後の一定値）で、発話区間が欠けずに含まれている
    assert np.ptp(audio[: pre_chunks * chunk_len]) < 1e-6
    assert (
        np.count_nonzero(
            audio[pre_chunks * chunk_len : (pre_chunks + speech_chunks) * chunk_len]
        )
        > 0
    )
    assert len(results) == 1
    print(f"[OK] 秒基準の発話前バッファのテスト成功 (長さ {len(audio)} サンプル = {len(audio)/sr:.1f}秒)")


if __name__ == "__main__":
    test_preprocess_audio()
    test_effective_threshold_adaptation()
    test_voiced_threshold_boundary()
    test_compute_boost_gain()
    test_format_elapsed_time()
    test_resample_audio()
    test_input_sample_rate_fallback()
    test_audio_queue_is_bounded()
    test_split_overlap_and_timestamps()
    test_pre_buffer_seconds_based()
    test_transient_noise_is_rejected()
    test_short_acknowledgement_is_kept()
    test_finalize_segment_min_duration_boundary()
    test_hard_split_tail_ignores_silence()
    test_speech_dropping_below_threshold_is_rejected()
    test_soft_split_on_short_pause()
    test_audio_processor_diarization()
    test_audio_processor_vad_and_short_speech()
    print("すべての音声前処理・プロセッサテストが正常に通過しました！")
