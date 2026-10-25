"""話者プロファイル更新ポリシーの単体テスト（モデル・ネットワーク非依存）"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.diarizer import SpeakerDiarizer, should_update_profile


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


def test_diarizer_policy_defaults():
    """過学習抑制のデフォルト値と初期状態"""
    diarizer = SpeakerDiarizer()
    assert not diarizer.is_loaded
    assert diarizer.num_speakers == 0
    assert diarizer.min_embedding_duration_sec == 0.8
    assert diarizer.profile_update_interval == 3
    assert diarizer.max_speakers == 8

    # パラメータ上書き
    custom = SpeakerDiarizer(
        min_embedding_duration_sec=1.2,
        profile_update_interval=5,
        max_speakers=4,
    )
    assert custom.min_embedding_duration_sec == 1.2
    assert custom.profile_update_interval == 5
    assert custom.max_speakers == 4
    print("[OK] 話者識別ポリシーのデフォルト値テスト成功")


if __name__ == "__main__":
    test_should_update_profile()
    test_diarizer_policy_defaults()
    print("すべての話者識別ポリシーテストが正常に通過しました！")
