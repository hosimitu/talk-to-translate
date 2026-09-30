"""UIと設定管理の統合テストスクリプト"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import ConfigManager
from src.ui import AppUI


def test_ui_config_persistence():
    """UIの初期値設定と設定変更による自動保存をテスト"""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_config_path = os.path.join(tmpdir, "config.json")
        
        # 事前にカスタム設定を用意
        initial_settings = {
            "transcription_model": "Whisper: small (軽量)",
            "translation_engine": "ローカル翻訳 (NLLB 1.3B)",
            "input_lang": "日本語 (ja)",
            "target_lang": "中国語 (zh)",
            "enable_separation": True,
            "mic_device": "テストマイク",
        }
        manager = ConfigManager(custom_path=fake_config_path)
        manager.save_config(initial_settings)

        # AppUI をインスタンス化し、custom_path の manager を差し替え
        app = AppUI()
        try:
            # ウィンドウを非表示
            app.withdraw()

            # 設定を再適用して検証
            app.config_manager = manager
            app.config = manager.load_config()

            # UIの選択肢を手動で反映テスト
            app.model_option.set("Whisper: small (軽量)")
            app.engine_option.set("ローカル翻訳 (NLLB 1.3B)")
            app.input_lang_option.set("日本語 (ja)")
            app.target_lang_option.set("中国語 (zh)")
            app.sep_checkbox.select()

            # 保存をトリガー
            app._save_current_settings()

            # ファイルから再度読み出して検証
            loaded = manager.load_config()
            assert loaded["transcription_model"] == "Whisper: small (軽量)"
            assert loaded["translation_engine"] == "ローカル翻訳 (NLLB 1.3B)"
            assert loaded["input_lang"] == "日本語 (ja)"
            assert loaded["target_lang"] == "中国語 (zh)"
            assert loaded["enable_separation"] is True
            print("[OK] UIと設定保存の統合テスト成功")
        finally:
            app.destroy()


def test_ui_text_styling():
    """テキストボックスのフォントサイズおよび行間設定をテスト"""
    app = AppUI()
    try:
        app.withdraw()
        # フォントサイズが 16 であること
        assert app.transcribe_textbox.cget("font").cget("size") == 16
        assert app.translate_textbox.cget("font").cget("size") == 16
        # 行間が設定されていること
        assert app.transcribe_textbox._textbox.cget("spacing2") == 4
        assert app.translate_textbox._textbox.cget("spacing2") == 4
        print("[OK] UIテキストスタイリング（フォントサイズ・行間）テスト成功")
    finally:
        app.destroy()


if __name__ == "__main__":
    test_ui_config_persistence()
    test_ui_text_styling()
