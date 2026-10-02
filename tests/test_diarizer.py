"""話者ダイアライザー (SpeakerDiarizer) の単体テストスクリプト"""

import huggingface_hub
import numpy as np
import soundfile as sf
from src.diarizer import SpeakerDiarizer


def test_speaker_diarizer_lifecycle():
    print("SpeakerDiarizer ライフサイクルおよび多話者識別のテスト実行中...")

    diarizer = SpeakerDiarizer(threshold=0.55)
    assert not diarizer.is_loaded
    assert diarizer.num_speakers == 0

    # 1. モデルロード
    diarizer.load_model()
    assert diarizer.is_loaded
    print("[OK] モデルのロード成功")

    # 2. 空音声のテスト
    res_empty = diarizer.identify_speaker(np.zeros(0, dtype=np.float32))
    assert res_empty is None
    print("[OK] 空音声の安全処理確認")

    # 3. 実データ（4人の話者音声）を用いたテスト
    wav_path = huggingface_hub.hf_hub_download(
        repo_id="csukuangfj/speaker-embedding-models",
        filename="0-four-speakers-zh.wav",
    )
    audio, sr = sf.read(wav_path, dtype="float32")

    # 話者1
    s1 = audio[int(sr * 0.5) : int(sr * 4.0)]
    spk1 = diarizer.identify_speaker(s1, sample_rate=sr)
    assert spk1 == "話者1", f"Expected 話者1, got {spk1}"
    print(f"[OK] 話者1の新規登録確認: {spk1}")

    # 話者2
    s2 = audio[int(sr * 6.0) : int(sr * 10.0)]
    spk2 = diarizer.identify_speaker(s2, sample_rate=sr)
    assert spk2 == "話者2", f"Expected 話者2, got {spk2}"
    print(f"[OK] 話者2の新規登録確認: {spk2}")

    # 話者3
    s3 = audio[int(sr * 12.0) : int(sr * 15.0)]
    spk3 = diarizer.identify_speaker(s3, sample_rate=sr)
    assert spk3 == "話者3", f"Expected 話者3, got {spk3}"
    print(f"[OK] 話者3の新規登録確認: {spk3}")

    # 再度話者1が発話（24秒付近）
    s1_again = audio[int(sr * 24.0) : int(sr * 27.0)]
    spk1_again = diarizer.identify_speaker(s1_again, sample_rate=sr)
    assert spk1_again == "話者1", f"Expected 話者1, got {spk1_again}"
    print(f"[OK] 話者1の再照合成功: {spk1_again}")

    assert diarizer.num_speakers == 3
    assert set(diarizer.all_speakers) == {"話者1", "話者2", "話者3"}

    # 4. リセットのテスト
    diarizer.reset()
    assert diarizer.num_speakers == 0
    assert len(diarizer.all_speakers) == 0
    print("[OK] リセット処理確認")


if __name__ == "__main__":
    test_speaker_diarizer_lifecycle()
    print("すべての SpeakerDiarizer テストが正常に通過しました！")
