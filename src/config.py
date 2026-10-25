"""
Talk-to-Translate 設定管理モジュール
UIの各種設定（モデル、言語、マイク、話者識別等）をJSONファイルとして永続化・復元します。
既定値と選択肢はこのモジュールを唯一の定義元とし、config.json は初回起動時に自動生成される
ローカル設定ファイル（Git管理外）です。
PyInstallerでEXE化された環境や管理者権限フォルダへの配置でも安全に動作するようフォールバック設計されています。
"""

import json
import os
import sys
import tempfile
from typing import Any, Dict


CONFIG_VERSION = 2  # v2: enable_separation を enable_diarization に統合

# プライバシー通知（クラウド翻訳時の外部送信）の同意バージョン。
# 文言を変更した場合はこの値を上げると、既存ユーザーにも再通知される。
PRIVACY_NOTICE_VERSION = 1

# UIの選択肢（config.json の値域とデフォルト値の唯一の定義元）
TRANSCRIPTION_MODELS = [
    "SenseVoice-Small (超高速・日中英推奨)",
    "Whisper: large-v3-turbo (高精度)",
    "Whisper: small (軽量)",
]

TRANSLATION_ENGINES = [
    "クラウド翻訳 (Google/MyMemory) (推奨)",
    "ローカル翻訳 (NLLB 600M) (推奨)",
    "ローカル翻訳 (NLLB 1.3B) (推奨)",
    "ローカル翻訳 (NLLB 3.3B)",
]

INPUT_LANGUAGES = ["自動検出 (auto)", "日本語 (ja)", "英語 (en)", "中国語 (zh)"]

TARGET_LANGUAGES = ["英語 (en)", "日本語 (ja)", "中国語 (zh)"]

DEFAULT_CONFIG: Dict[str, Any] = {
    "config_version": CONFIG_VERSION,
    "transcription_model": TRANSCRIPTION_MODELS[0],
    "translation_engine": TRANSLATION_ENGINES[0],
    "input_lang": INPUT_LANGUAGES[0],
    "target_lang": TARGET_LANGUAGES[0],
    "enable_diarization": False,
    "mic_device": "",
    "privacy_ack_version": 0,
}

# 値の妥当性検証に使う選択肢一覧（DEFAULT_CONFIG がここに含まれることをテストで保証）
CONFIG_SCHEMA: Dict[str, list] = {
    "transcription_model": TRANSCRIPTION_MODELS,
    "translation_engine": TRANSLATION_ENGINES,
    "input_lang": INPUT_LANGUAGES,
    "target_lang": TARGET_LANGUAGES,
}


def _as_text(value: Any) -> str:
    """設定値を安全に文字列化する（文字列以外は空文字扱い）"""
    return value.strip() if isinstance(value, str) else ""


def normalize_transcription_model(value: Any) -> str:
    """文字起こしモデルの設定値を正規化（旧表記・部分一致にも対応）"""
    text = _as_text(value)
    if text in TRANSCRIPTION_MODELS:
        return text

    lowered = text.lower()
    if "sensevoice" in lowered:
        return TRANSCRIPTION_MODELS[0]
    if "small" in lowered:
        return TRANSCRIPTION_MODELS[2]
    if "whisper" in lowered or "turbo" in lowered:
        return TRANSCRIPTION_MODELS[1]
    return DEFAULT_CONFIG["transcription_model"]


def normalize_translation_engine(value: Any) -> str:
    """翻訳エンジンの設定値を正規化（旧表記の「(推奨)」なし等に対応）"""
    text = _as_text(value)
    if text in TRANSLATION_ENGINES:
        return text

    if "ローカル" in text:
        if "1.3B" in text:
            return TRANSLATION_ENGINES[2]
        if "3.3B" in text:
            return TRANSLATION_ENGINES[3]
        return TRANSLATION_ENGINES[1]
    if "クラウド" in text or "cloud" in text.lower():
        return TRANSLATION_ENGINES[0]
    return DEFAULT_CONFIG["translation_engine"]


def normalize_input_language(value: Any) -> str:
    """入力言語の設定値を正規化（言語コードのみの旧表記にも対応）"""
    text = _as_text(value)
    if text in INPUT_LANGUAGES:
        return text

    if "ja" in text or "日本" in text:
        return INPUT_LANGUAGES[1]
    if "zh" in text or "中国" in text or "中文" in text:
        return INPUT_LANGUAGES[3]
    if "en" in text or "英語" in text:
        return INPUT_LANGUAGES[2]
    return DEFAULT_CONFIG["input_lang"]


def normalize_target_language(value: Any) -> str:
    """翻訳先言語の設定値を正規化（言語コードのみの旧表記にも対応）"""
    text = _as_text(value)
    if text in TARGET_LANGUAGES:
        return text

    if "ja" in text or "日本" in text:
        return TARGET_LANGUAGES[1]
    if "zh" in text or "中国" in text or "中文" in text:
        return TARGET_LANGUAGES[2]
    if "en" in text or "英語" in text:
        return TARGET_LANGUAGES[0]
    return DEFAULT_CONFIG["target_lang"]


def normalize_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    読み込んだ設定を検証し、既定のキーだけを持つ正規化済みの辞書を返す。
    不正値・旧形式の値はデフォルト値へ、未知のキーは破棄する。
    """
    if not isinstance(data, dict):
        data = {}

    config = DEFAULT_CONFIG.copy()
    config["transcription_model"] = normalize_transcription_model(data.get("transcription_model"))
    config["translation_engine"] = normalize_translation_engine(data.get("translation_engine"))
    config["input_lang"] = normalize_input_language(data.get("input_lang"))
    config["target_lang"] = normalize_target_language(data.get("target_lang"))

    enable_diarization = data.get("enable_diarization")
    if not isinstance(enable_diarization, bool):
        # 旧バージョン（v1）のキー名との互換性
        enable_diarization = data.get("enable_separation")
    if isinstance(enable_diarization, bool):
        config["enable_diarization"] = enable_diarization

    mic_device = data.get("mic_device")
    if isinstance(mic_device, str):
        config["mic_device"] = mic_device

    privacy_ack = data.get("privacy_ack_version")
    if isinstance(privacy_ack, bool) or not isinstance(privacy_ack, int) or privacy_ack < 0:
        privacy_ack = 0
    config["privacy_ack_version"] = privacy_ack

    config["config_version"] = CONFIG_VERSION
    return config


def _write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    """JSON を一時ファイル経由でアトミックに書き込む（破損防止）"""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".config-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


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
                    # 不正値・旧形式の値をデフォルトへ補正して返す
                    config = normalize_config(data)
        except Exception as e:
            print(f"[ConfigManager] 設定ファイルの読み込みに失敗しました ({e})。デフォルト値を使用します。")

        return config

    def save_config(self, config_data: Dict[str, Any]) -> bool:
        """設定データをJSONファイルへ保存する（保存前に値域を検証）"""
        path = self.get_config_path()
        # 未知のキーを落とし、不正値をデフォルトへ補正してから保存する
        data_to_save = normalize_config(config_data)

        try:
            _write_json_atomic(path, data_to_save)
            return True
        except (OSError, PermissionError) as e:
            print(f"[ConfigManager] 主パスへの設定保存に失敗しました ({e})。AppDataへのフォールバックを試みます。")
            try:
                fallback_path = os.path.join(self.get_appdata_dir(), self.FILENAME)
                _write_json_atomic(fallback_path, data_to_save)
                self._config_path = fallback_path
                return True
            except Exception as e_fallback:
                print(f"[ConfigManager] フォールバック設定保存にも失敗しました: {e_fallback}")
                return False
        except Exception as e:
            print(f"[ConfigManager] 設定の保存中に予期せぬエラーが発生しました: {e}")
            return False
