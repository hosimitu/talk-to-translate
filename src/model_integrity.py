"""
Hugging Face Hub から取得するモデルファイルの完全性検証。

上流リポジトリの main は差し替えられ得るため、commit SHA で revision を固定し、
取得したファイルの SHA256 を照合してからネイティブ実行系（sherpa-onnx など）へ渡す。
"""

import hashlib
import re

import huggingface_hub


# revision は commit SHA（40桁の16進）のみ許可する（main やタグの可変参照を拒否）
_PINNED_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


def compute_sha256(path) -> str:
    """ファイル全体の SHA256（16進小文字）を返す"""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(
    repo_id: str,
    filename: str,
    revision: str,
    expected_sha256: str,
) -> str:
    """
    モデルファイルを固定 revision で取得し、SHA256 を照合してパスを返す。

    revision が commit SHA でない場合、またはハッシュが一致しない場合は
    改ざん・取り違えとみなして例外を送出し、呼び出し元へパスを渡さない。
    """
    if not _PINNED_REVISION_PATTERN.fullmatch(str(revision)):
        raise ValueError(
            f"モデルの revision は commit SHA で固定してください: {revision!r}"
        )

    path = huggingface_hub.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
    )
    actual = compute_sha256(path)
    if actual != expected_sha256.lower():
        raise RuntimeError(
            f"モデルファイルのSHA256が一致しません: {filename} "
            f"(expected={expected_sha256}, actual={actual})"
        )
    return path
