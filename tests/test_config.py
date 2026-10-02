"""設定管理モジュールの単体テストスクリプト"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import ConfigManager, DEFAULT_CONFIG


def test_default_config():
    """設定ファイルが存在しない場合のデフォルト値確認"""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_config_path = os.path.join(tmpdir, "config.json")
        manager = ConfigManager(custom_path=fake_config_path)
        loaded = manager.load_config()

        assert loaded == DEFAULT_CONFIG
        assert loaded["transcription_model"] == "SenseVoice-Small (超高速・日中英推奨)"
        assert loaded["translation_engine"] == "クラウド翻訳 (Google/MyMemory) (推奨)"
        print("[OK] デフォルト設定の読み込みテスト成功")


def test_save_and_load_config():
    """設定の保存と再読み込みテスト"""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_config_path = os.path.join(tmpdir, "config.json")
        manager = ConfigManager(custom_path=fake_config_path)

        new_settings = {
            "transcription_model": "Whisper: large-v3-turbo (高精度)",
            "translation_engine": "ローカル翻訳 (NLLB 600M)",
            "input_lang": "日本語 (ja)",
            "target_lang": "英語 (en)",
            "enable_separation": True,
            "mic_device": "1: USB Microphone",
        }

        success = manager.save_config(new_settings)
        assert success is True
        assert os.path.exists(fake_config_path)

        loaded = manager.load_config()
        assert loaded["transcription_model"] == "Whisper: large-v3-turbo (高精度)"
        assert loaded["translation_engine"] == "ローカル翻訳 (NLLB 600M)"
        assert loaded["input_lang"] == "日本語 (ja)"
        assert loaded["target_lang"] == "英語 (en)"
        assert loaded["enable_separation"] is True
        assert loaded["mic_device"] == "1: USB Microphone"
        print("[OK] 設定の保存・再読み込みテスト成功")


def test_corrupt_config_fallback():
    """破損したJSONファイル時のフォールバック動作テスト"""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_config_path = os.path.join(tmpdir, "config.json")
        with open(fake_config_path, "w", encoding="utf-8") as f:
            f.write("{ corrupt json invalid ...")

        manager = ConfigManager(custom_path=fake_config_path)
        loaded = manager.load_config()

        # 破損していてもデフォルト値が安全に返ること
        assert loaded == DEFAULT_CONFIG
        print("[OK] 破損JSONファイルの安全なフォールバックテスト成功")


def test_partial_config_merge():
    """一部のキーのみ含まれる場合、未設定キーはデフォルト値が維持されるか"""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_config_path = os.path.join(tmpdir, "config.json")
        manager = ConfigManager(custom_path=fake_config_path)

        partial_settings = {
            "transcription_model": "Whisper: small (軽量)",
        }
        manager.save_config(partial_settings)

        loaded = manager.load_config()
        assert loaded["transcription_model"] == "Whisper: small (軽量)"
        assert loaded["translation_engine"] == DEFAULT_CONFIG["translation_engine"]
        assert loaded["target_lang"] == DEFAULT_CONFIG["target_lang"]
        print("[OK] 部分設定のマージテスト成功")


if __name__ == "__main__":
    test_default_config()
    test_save_and_load_config()
    test_corrupt_config_fallback()
    test_partial_config_merge()
    print("すべての設定管理テストが正常に通過しました！")
