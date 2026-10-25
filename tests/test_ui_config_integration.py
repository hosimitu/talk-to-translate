"""UIと設定管理の統合テストスクリプト"""

import os
import queue
import sys
import tempfile
import tkinter as tk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import PRIVACY_NOTICE_VERSION, ConfigManager
from src.ui import AppUI


def test_ui_config_persistence():
    """UIの初期値設定と設定変更による自動保存をテスト"""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_config_path = os.path.join(tmpdir, "config.json")
        
        # 事前にカスタム設定を用意
        initial_settings = {
            "transcription_model": "Whisper: small (軽量)",
            "translation_engine": "ローカル翻訳 (NLLB 1.3B) (推奨)",
            "input_lang": "日本語 (ja)",
            "target_lang": "中国語 (zh)",
            "enable_diarization": True,
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
            app.engine_option.set("ローカル翻訳 (NLLB 1.3B) (推奨)")
            app.input_lang_option.set("日本語 (ja)")
            app.target_lang_option.set("中国語 (zh)")
            app.diarization_checkbox.select()

            # 保存をトリガー
            app._save_current_settings()

            # ファイルから再度読み出して検証
            loaded = manager.load_config()
            assert loaded["transcription_model"] == "Whisper: small (軽量)"
            assert loaded["translation_engine"] == "ローカル翻訳 (NLLB 1.3B) (推奨)"
            assert loaded["input_lang"] == "日本語 (ja)"
            assert loaded["target_lang"] == "中国語 (zh)"
            assert loaded["enable_diarization"] is True
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


def test_p1_translation_pipeline_and_privacy():
    """翻訳の順序保証・世代破棄・言語引き継ぎ・プライバシー表示のテスト"""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_config_path = os.path.join(tmpdir, "config.json")
        manager = ConfigManager(custom_path=fake_config_path)

        app = AppUI()
        try:
            app.withdraw()
            app.config_manager = manager
            app.config = manager.load_config()

            # 1. プライバシーバッジが翻訳エンジンに追従する
            app.engine_option.set("クラウド翻訳 (Google/MyMemory) (推奨)")
            app._update_privacy_badge()
            assert "外部送信: 有効" in app.privacy_badge.cget("text")
            app.engine_option.set("ローカル翻訳 (NLLB 600M) (推奨)")
            app._update_privacy_badge()
            assert "ローカルのみ" in app.privacy_badge.cget("text")

            # 1b. モデルバッジがモデル選択と推論デバイスに追従する
            app.model_option.set("SenseVoice-Small (超高速・日中英推奨)")
            app._update_model_badge()
            assert "SenseVoice" in app.model_badge.cget("text")
            app.model_option.set("Whisper: small (軽量)")
            app._update_model_badge()
            assert "Whisper" in app.model_badge.cget("text")
            assert any(label in app.model_badge.cget("text") for label in ("GPU", "CPU"))

            # 2. 同意するとバージョンが保存され、バナーが閉じる
            app.config["privacy_ack_version"] = 0
            app._refresh_privacy_notice()
            assert app.privacy_notice.winfo_manager() == "grid"
            app._acknowledge_privacy_notice()
            assert manager.load_config()["privacy_ack_version"] == PRIVACY_NOTICE_VERSION
            assert app.privacy_notice.winfo_manager() == ""

            # 3. 古い世代の翻訳結果は破棄され、新しいものだけ反映される
            app.translate_textbox.delete("1.0", tk.END)
            app._translation_generation = 3
            app.translation_queue.put((2, "古い翻訳"))
            app.translation_queue.put((3, "新しい翻訳"))
            app._drain_translation_queue()
            assert app.translate_textbox.get("1.0", tk.END).strip() == "新しい翻訳"

            # 4. 文字起こしキュー経由で検出言語が記録される
            app.transcribe_textbox.delete("1.0", tk.END)
            app.transcription_queue.put(("こんにちは", "ja"))
            app._drain_transcription_queue()
            assert app._last_source_lang == "ja"
            assert "こんにちは" in app.transcribe_textbox.get("1.0", tk.END)

            # 5. 自動検出時は検出言語が翻訳ジョブへ引き継がれる
            captured: queue.Queue = queue.Queue()
            original_queue = app._translation_requests
            app._translation_requests = captured
            try:
                app._source_lang = "auto"
                app._target_lang = "en"
                app._last_source_lang = "ja"
                app._translation_buffer.append("[話者1] こんにちは")
                with app._buffer_lock:
                    app._flush_translation_buffer_locked()
                generation, lines, target, source = captured.get_nowait()
                assert generation == app._translation_generation
                assert source == "ja" and target == "en"
                assert lines == ["[話者1] こんにちは"]
            finally:
                app._translation_requests = original_queue

            # 6. 全消去で世代が進み、未反映の翻訳結果も破棄される
            generation_before = app._translation_generation
            app.translation_queue.put((generation_before, "消えるはずの翻訳"))
            app._clear_all_text()
            assert app._translation_generation == generation_before + 1
            assert app.translation_queue.empty()

            print("[OK] 翻訳パイプライン・プライバシー表示テスト成功")
        finally:
            app.destroy()


if __name__ == "__main__":
    test_ui_config_persistence()
    test_ui_text_styling()
    test_p1_translation_pipeline_and_privacy()
