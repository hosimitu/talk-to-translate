"""
Talk-to-Translate EXE ビルドスクリプト (v1.2.0)
PyInstaller を使用して、NLLB-200ローカル翻訳、MossFormer2、SenseVoice/Whisperを含むスタンドアロン Windows 実行可能ファイル (.exe) を構築します。
"""

import os
import subprocess
import sys


def build():
    print("========================================")
    print("Talk-to-Translate EXE ビルドを開始します (v1.2.0)")
    print("========================================")

    pyinstaller_cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onedir",                # 起動が最も高速で安定するフォルダ形式
        "--windowed",              # 黒いコマンドプロンプト画面を表示しない
        "--name", "Talk-to-Translate",
        # 静的アセット・テーマ・設定の同梱
        "--collect-data", "customtkinter",
        "--collect-data", "sherpa_onnx",
        "--collect-data", "clearvoice",
        "--collect-data", "transformers",
        # C++ / ONNX / Torch 関連DLLの同梱
        "--collect-binaries", "sherpa_onnx",
        "--collect-binaries", "sherpa_onnx_core",
        "--collect-binaries", "ctranslate2",
        "--collect-binaries", "sounddevice",
        # 動的インポートされるサブモジュールの同梱
        "--collect-submodules", "deep_translator",
        "--collect-submodules", "faster_whisper",
        "--collect-submodules", "sherpa_onnx",
        "--collect-submodules", "huggingface_hub",
        "--collect-submodules", "clearvoice",
        "--collect-submodules", "soundfile",
        "--collect-submodules", "transformers",
        # エントリーポイント
        "main.py",
    ]

    print("実行コマンド:", " ".join(pyinstaller_cmd))
    res = subprocess.run(pyinstaller_cmd)

    if res.returncode != 0:
        print("\n[エラー] ビルドに失敗しました。")
        sys.exit(res.returncode)

    output_dir = os.path.abspath(os.path.join("dist", "Talk-to-Translate"))
    exe_path = os.path.join(output_dir, "Talk-to-Translate.exe")

    print("\n========================================")
    print("[OK] ビルドが正常に完了しました！")
    print(f"出力フォルダ: {output_dir}")
    print(f"実行可能ファイル: {exe_path}")
    print("========================================")


if __name__ == "__main__":
    build()
