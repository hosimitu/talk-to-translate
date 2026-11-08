"""話者プロファイル更新ポリシーの単体テスト（モデル・ネットワーク非依存）"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.diarizer import SpeakerDiarizer, extract_embedding_window, should_update_profile


def test_should_update_profile():
    """プロファイル更新が一定間隔に間引かれること"""
    assert should_update_profile(1, 3) is False
    assert should_update_profile(2, 3) is False
    assert should_update_profile(3, 3) is True
    assert should_update_profile(4, 3) is False
    assert should_update_profile(6, 3) is True

    # 間隔1以下なら毎回更新
    assert should_update_profile(1, 1) is True
    assert should_update_profile(5, 0) is True

    # 0回以下は更新しない
    assert should_update_profile(0, 3) is False
    assert should_update_profile(-1, 3) is False
    print("[OK] 話者プロファイル更新間隔のテスト成功")


def test_extract_embedding_window():
    """埋め込み計算に使う区間が最大長に切り出され、最もエネルギーの高い部分が選ばれること"""
    sr = 16000
    max_sec = 3.5
    max_samples = int(round(max_sec * sr))

    # 最大長以下の音声はそのまま返る
    short = np.ones(sr * 2, dtype=np.float32) * 0.1
    out_short = extract_embedding_window(short, sr, max_duration_sec=max_sec)
    assert len(out_short) == len(short)
    assert np.array_equal(out_short, short)

    # 空音声・不正値でも安全
    empty = np.zeros(0, dtype=np.float32)
    assert len(extract_embedding_window(empty, sr)) == 0
    assert np.array_equal(
        extract_embedding_window(short, 0, max_duration_sec=max_sec), short
    )
    assert np.array_equal(
        extract_embedding_window(short, sr, max_duration_sec=0), short
    )

    # 8秒の音声: 先頭と末尾は無音、3〜5秒だけ振幅が大きい
    total_sec = 8.0
    audio = np.zeros(int(sr * total_sec), dtype=np.float32)
    loud_start = int(sr * 3.0)
    loud_end = int(sr * 5.0)
    audio[loud_start:loud_end] = 0.5

    window = extract_embedding_window(audio, sr, max_duration_sec=max_sec)
    # 長さは最大長ちょうど
    assert len(window) == max_samples
    # 選ばれた区間の中に有音部が十分含まれていること（無音だけの区間は選ばれない）
    loud_in_window = int(np.count_nonzero(window))
    assert loud_in_window >= int((loud_end - loud_start) * 0.9), (
        f"有音部が十分に含まれていません: {loud_in_window} サンプル"
    )
    print(f"[OK] 埋め込み窓の切り出しテスト成功 (有音 {loud_in_window} サンプルを保持)")


def test_diarizer_policy_defaults():
    """過学習抑制のデフォルト値と初期状態"""
    diarizer = SpeakerDiarizer()
    assert not diarizer.is_loaded
    assert diarizer.num_speakers == 0
    assert diarizer.min_embedding_duration_sec == 0.8
    assert diarizer.profile_update_interval == 3
    assert diarizer.max_speakers == 8
    # 埋め込み計算の音声長上限（長い発話での推論時間を抑える）
    assert diarizer.max_embedding_duration_sec == 3.5
    # スレッド数はCPUコア数から自動決定される（低スペックPCを圧迫しない範囲）
    assert 1 <= diarizer.num_threads <= 4

    # パラメータ上書き
    custom = SpeakerDiarizer(
        min_embedding_duration_sec=1.2,
        profile_update_interval=5,
        max_speakers=4,
        max_embedding_duration_sec=2.0,
        num_threads=3,
    )
    assert custom.min_embedding_duration_sec == 1.2
    assert custom.profile_update_interval == 5
    assert custom.max_speakers == 4
    assert custom.max_embedding_duration_sec == 2.0
    assert custom.num_threads == 3
    print("[OK] 話者識別ポリシーのデフォルト値テスト成功")


if __name__ == "__main__":
    test_should_update_profile()
    test_extract_embedding_window()
    test_diarizer_policy_defaults()
    print("すべての話者識別ポリシーテストが正常に通過しました！")
