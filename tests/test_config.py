"""設定管理モジュールの単体テストスクリプト"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import (
    CONFIG_SCHEMA,
    CONFIG_VERSION,
    DEFAULT_CONFIG,
    ConfigManager,
)


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


def test_default_config_is_valid():
    """デフォルト値がUI選択肢（スキーマ）に含まれていること"""
    for key, allowed in CONFIG_SCHEMA.items():
        assert DEFAULT_CONFIG[key] in allowed, f"{key} のデフォルト値が選択肢にありません"
    assert DEFAULT_CONFIG["config_version"] == CONFIG_VERSION
    print("[OK] デフォルト設定の整合性テスト成功")


def test_save_and_load_config():
    """設定の保存と再読み込みテスト"""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_config_path = os.path.join(tmpdir, "config.json")
        manager = ConfigManager(custom_path=fake_config_path)

        new_settings = {
            "transcription_model": "Whisper: large-v3-turbo (高精度)",
            "translation_engine": "ローカル翻訳 (NLLB 600M) (推奨)",
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
        assert loaded["translation_engine"] == "ローカル翻訳 (NLLB 600M) (推奨)"
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


def test_invalid_values_fall_back():
    """不正な型・未知の値・未知のキーがデフォルトへ補正されること"""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_config_path = os.path.join(tmpdir, "config.json")
        invalid_settings = {
            "transcription_model": "存在しないモデル",
            "translation_engine": 123,
            "input_lang": ["日本語 (ja)"],
            "target_lang": "Klingon",
            "enable_separation": "yes",
            "mic_device": None,
            "unknown_key": "value",
        }
        with open(fake_config_path, "w", encoding="utf-8") as f:
            json.dump(invalid_settings, f, ensure_ascii=False)

        loaded = ConfigManager(custom_path=fake_config_path).load_config()

        assert loaded == DEFAULT_CONFIG
        assert "unknown_key" not in loaded
        print("[OK] 不正値のデフォルト補正テスト成功")


def test_legacy_values_are_migrated():
    """旧バージョンの表記（(推奨)なし・言語コードのみ等）が正規化されること"""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_config_path = os.path.join(tmpdir, "config.json")
        legacy_settings = {
            "transcription_model": "SenseVoice",
            "translation_engine": "ローカル翻訳 (NLLB 600M)",
            "input_lang": "ja",
            "target_lang": "日本語",
        }
        with open(fake_config_path, "w", encoding="utf-8") as f:
            json.dump(legacy_settings, f, ensure_ascii=False)

        loaded = ConfigManager(custom_path=fake_config_path).load_config()

        assert loaded["transcription_model"] == "SenseVoice-Small (超高速・日中英推奨)"
        assert loaded["translation_engine"] == "ローカル翻訳 (NLLB 600M) (推奨)"
        assert loaded["input_lang"] == "日本語 (ja)"
        assert loaded["target_lang"] == "日本語 (ja)"
        print("[OK] 旧表記のマイグレーションテスト成功")


def test_save_normalizes_values():
    """保存時にも値域の検証・正規化が行われること"""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_config_path = os.path.join(tmpdir, "config.json")
        manager = ConfigManager(custom_path=fake_config_path)

        manager.save_config({"translation_engine": "ローカル翻訳 (NLLB 1.3B)"})

        with open(fake_config_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["translation_engine"] == "ローカル翻訳 (NLLB 1.3B) (推奨)"
        assert saved["config_version"] == CONFIG_VERSION
        print("[OK] 保存時の正規化テスト成功")


if __name__ == "__main__":
    test_default_config()
    test_default_config_is_valid()
    test_save_and_load_config()
    test_corrupt_config_fallback()
    test_partial_config_merge()
    test_invalid_values_fall_back()
    test_legacy_values_are_migrated()
    test_save_normalizes_values()
    print("すべての設定管理テストが正常に通過しました！")
