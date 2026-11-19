"""モデルファイルの revision 固定・SHA256 照合の単体テスト（ネットワーク非依存）"""

import hashlib
import os
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.diarizer import SpeakerDiarizer
from src.model_integrity import compute_sha256, download_verified
from src.sensevoice_engine import SenseVoiceEngine


PINNED_REVISION = "0" * 40


def test_compute_sha256():
    """実ファイルの SHA256 が hashlib の結果と一致すること"""
    data = Path(__file__).read_bytes()
    assert compute_sha256(__file__) == hashlib.sha256(data).hexdigest()
    print("[OK] SHA256 計算テスト成功")


def test_download_verified():
    """固定 revision での取得と SHA256 照合、不一致・可変参照の拒否"""
    expected = "a" * 64
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return "/models/model.onnx"

    with patch(
        "src.model_integrity.huggingface_hub.hf_hub_download",
        side_effect=fake_download,
    ), patch("src.model_integrity.compute_sha256", return_value=expected):
        result = download_verified("org/repo", "model.onnx", PINNED_REVISION, expected)
    assert result == "/models/model.onnx"
    assert calls == [
        {
            "repo_id": "org/repo",
            "filename": "model.onnx",
            "revision": PINNED_REVISION,
        }
    ]

    # ハッシュ不一致は取得後に拒否する
    with patch(
        "src.model_integrity.huggingface_hub.hf_hub_download",
        return_value="/models/model.onnx",
    ), patch("src.model_integrity.compute_sha256", return_value="0" * 64):
        try:
            download_verified("org/repo", "model.onnx", PINNED_REVISION, expected)
        except RuntimeError as e:
            assert "SHA256" in str(e)
        else:
            raise AssertionError("SHA256 不一致を拒否していません")

    # main などの可変参照は取得前に拒否する
    with patch(
        "src.model_integrity.huggingface_hub.hf_hub_download"
    ) as download, patch("src.model_integrity.compute_sha256", return_value=expected):
        try:
            download_verified("org/repo", "model.onnx", "main", expected)
        except ValueError:
            pass
        else:
            raise AssertionError("revision=main を拒否していません")
        download.assert_not_called()

    print("[OK] モデル取得の revision 固定・SHA256 照合テスト成功")


def test_model_revisions_are_pinned():
    """モデル取得の revision と SHA256 が固定されていること"""
    assert re.fullmatch(r"[0-9a-f]{40}", SenseVoiceEngine.MODEL_REVISION)
    assert re.fullmatch(r"[0-9a-f]{64}", SenseVoiceEngine.MODEL_SHA256)
    assert re.fullmatch(r"[0-9a-f]{64}", SenseVoiceEngine.TOKENS_SHA256)

    assert re.fullmatch(r"[0-9a-f]{40}", SpeakerDiarizer.MODEL_REVISION)
    assert set(SpeakerDiarizer.MODEL_SHA256_BY_FILENAME) == {
        SpeakerDiarizer.DEFAULT_MODEL
    }
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        SpeakerDiarizer.MODEL_SHA256_BY_FILENAME[SpeakerDiarizer.DEFAULT_MODEL],
    )
    print("[OK] モデル revision/SHA256 の固定テスト成功")


def test_sensevoice_uses_verified_download():
    """SenseVoice が固定 revision とハッシュでモデルを取得すること"""
    engine = SenseVoiceEngine.__new__(SenseVoiceEngine)
    engine.num_threads = 2
    calls = []

    def fake_download(repo_id, filename, revision, expected_sha256):
        calls.append((repo_id, filename, revision, expected_sha256))
        return f"/models/{filename}"

    with patch(
        "src.sensevoice_engine.download_verified", side_effect=fake_download
    ), patch(
        "src.sensevoice_engine.sherpa_onnx.OfflineRecognizer.from_sense_voice",
        return_value=MagicMock(),
    ):
        engine._build_recognizer("ja")

    assert calls == [
        (
            SenseVoiceEngine.REPO_ID,
            SenseVoiceEngine.MODEL_FILENAME,
            SenseVoiceEngine.MODEL_REVISION,
            SenseVoiceEngine.MODEL_SHA256,
        ),
        (
            SenseVoiceEngine.REPO_ID,
            SenseVoiceEngine.TOKENS_FILENAME,
            SenseVoiceEngine.MODEL_REVISION,
            SenseVoiceEngine.TOKENS_SHA256,
        ),
    ]
    print("[OK] SenseVoice の検証済みモデル取得テスト成功")


def test_diarizer_uses_verified_download():
    """話者埋め込みモデルも固定 revision とハッシュで取得すること"""
    diarizer = SpeakerDiarizer(num_threads=1)
    calls = []

    def fake_download(repo_id, filename, revision, expected_sha256):
        calls.append((repo_id, filename, revision, expected_sha256))
        return "/models/speaker.onnx"

    config = MagicMock()
    config.validate.return_value = True
    extractor = MagicMock()
    extractor.dim = 192

    with patch("src.diarizer.download_verified", side_effect=fake_download), patch(
        "src.diarizer.sherpa_onnx.SpeakerEmbeddingExtractorConfig",
        return_value=config,
    ), patch(
        "src.diarizer.sherpa_onnx.SpeakerEmbeddingExtractor",
        return_value=extractor,
    ), patch(
        "src.diarizer.sherpa_onnx.SpeakerEmbeddingManager",
        return_value=MagicMock(),
    ):
        diarizer.load_model()

    assert diarizer.is_loaded
    assert calls == [
        (
            SpeakerDiarizer.REPO_ID,
            SpeakerDiarizer.DEFAULT_MODEL,
            SpeakerDiarizer.MODEL_REVISION,
            SpeakerDiarizer.MODEL_SHA256_BY_FILENAME[SpeakerDiarizer.DEFAULT_MODEL],
        )
    ]

    # 未検証のモデルファイル名は取得前に拒否する
    unknown = SpeakerDiarizer(model_filename="unknown.onnx", num_threads=1)
    with patch("src.diarizer.download_verified") as download:
        try:
            unknown.load_model()
        except RuntimeError as e:
            assert "検証済みでない" in str(e)
        else:
            raise AssertionError("未検証モデルを拒否していません")
        download.assert_not_called()
    print("[OK] 話者埋め込みモデルの検証済み取得テスト成功")


if __name__ == "__main__":
    test_compute_sha256()
    test_download_verified()
    test_model_revisions_are_pinned()
    test_sensevoice_uses_verified_download()
    test_diarizer_uses_verified_download()
    print("すべてのモデル完全性検証テストが正常に通過しました！")
