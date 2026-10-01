"""
Talk-to-Translate 設定管理モジュール
UIの各種設定（モデル、言語、マイク、音源分離等）をJSONファイルとして永続化・復元します。
PyInstallerでEXE化された環境や管理者権限フォルダへの配置でも安全に動作するようフォールバック設計されています。
"""

import json
import os
import sys
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "transcription_model": "SenseVoice-Small (超高速・日中英推奨)",
    "translation_engine": "クラウド翻訳 (Google/MyMemory) (推奨)",
    "input_lang": "自動検出 (auto)",
    "target_lang": "英語 (en)",
    "enable_separation": False,
    "mic_device": "",
}


class ConfigManager:
    """アプリケーション設定の読み込み・保存を管理するクラス"""

    FILENAME = "config.json"

    def __init__(self, custom_path: str | None = None):
        self.custom_path = custom_path
        self._config_path: str | None = None

    def get_base_dir(self) -> str:
        """実行環境（EXEかスクリプトか）に応じたベースディレクトリを取得"""
        if getattr(sys, "frozen", False):
            # PyInstaller による EXE 実行時
            return os.path.dirname(sys.executable)
        else:
            # Python スクリプト実行時（プロジェクトルート）
            return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def get_appdata_dir(self) -> str:
        """AppData（またはホームディレクトリ）配下の設定ディレクトリを取得・作成"""
        appdata = os.environ.get("APPDATA")
        if appdata:
            path = os.path.join(appdata, "Talk-to-Translate")
        else:
            path = os.path.join(os.path.expanduser("~"), ".talk-to-translate")
        os.makedirs(path, exist_ok=True)
        return path

    def get_config_path(self) -> str:
        """利用可能な設定ファイルパスを決定して返す"""
        if self.custom_path:
            return self.custom_path

        if self._config_path:
            return self._config_path

        # 1. ベースディレクトリ（exeの隣またはプロジェクトルート）を優先試行
        base_dir = self.get_base_dir()
        candidate_path = os.path.join(base_dir, self.FILENAME)

        # 既にベースディレクトリに存在する場合はそれを採用
        if os.path.exists(candidate_path):
            self._config_path = candidate_path
            return self._config_path

        # ベースディレクトリに書き込み可能かチェック（テスト書き込み）
        try:
            test_file = os.path.join(base_dir, f".write_test_{os.getpid()}.tmp")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(test_file)
            self._config_path = candidate_path
            return self._config_path
        except (OSError, PermissionError):
            # 書き込み権限がない場合（例: C:\Program Files 配下に配置されたEXE等）
            pass

        # 2. AppData ディレクトリへフォールバック
        appdata_dir = self.get_appdata_dir()
        self._config_path = os.path.join(appdata_dir, self.FILENAME)
        return self._config_path

    def load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込み、辞書として返す。存在しない・壊れている場合はデフォルト値を返す"""
        config = DEFAULT_CONFIG.copy()
        path = self.get_config_path()

        if not os.path.exists(path):
            return config

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # 既存のデフォルト値にロードしたデータをマージ
                    config.update(data)
        except Exception as e:
            print(f"[ConfigManager] 設定ファイルの読み込みに失敗しました ({e})。デフォルト値を使用します。")

        return config

    def save_config(self, config_data: Dict[str, Any]) -> bool:
        """設定データをJSONファイルへ保存する"""
        path = self.get_config_path()
        data_to_save = DEFAULT_CONFIG.copy()
        data_to_save.update(config_data)

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            return True
        except (OSError, PermissionError) as e:
            print(f"[ConfigManager] 主パスへの設定保存に失敗しました ({e})。AppDataへのフォールバックを試みます。")
            try:
                fallback_path = os.path.join(self.get_appdata_dir(), self.FILENAME)
                with open(fallback_path, "w", encoding="utf-8") as f:
                    json.dump(data_to_save, f, ensure_ascii=False, indent=2)
                self._config_path = fallback_path
                return True
            except Exception as e_fallback:
                print(f"[ConfigManager] フォールバック設定保存にも失敗しました: {e_fallback}")
                return False
        except Exception as e:
            print(f"[ConfigManager] 設定の保存中に予期せぬエラーが発生しました: {e}")
            return False
