"""
Talk-to-Translate アプリケーション エントリーポイント
"""

import os
import sys
import multiprocessing

# --- Windows GUI (console=False / windowed) 環境下での標準ストリーム保護 ---
class _SafeStream:
    """NoneType の標準出力を安全に置き換え、ログファイルへの出力または破棄を行うストリーム"""

    def __init__(self, log_file=None):
        self._log_file = log_file

    def write(self, s: str) -> int:
        if self._log_file:
            try:
                self._log_file.write(s)
                self._log_file.flush()
            except Exception:
                pass
        return len(s) if s else 0

    def flush(self) -> None:
        if self._log_file:
            try:
                self._log_file.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return False


class _SafeReader:
    """NoneType の標準入力を安全に置き換えるダミーストリーム"""

    def read(self, *args, **kwargs) -> str:
        return ""

    def readline(self, *args, **kwargs) -> str:
        return ""

    def readlines(self, *args, **kwargs) -> list:
        return []


def _setup_safe_environment():
    """GUI起動時に標準ストリームがNoneになる問題や進捗バーのクラッシュを防止"""
    # Hugging Face Hub / ModelScope の進捗バーによるGUI環境下でのエラーを防止
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("MODELSCOPE_DISABLE_PROGRESS_BAR", "1")

    log_file = None
    if sys.stdout is None or sys.stderr is None:
        try:
            appdata = os.environ.get("APPDATA")
            if appdata:
                log_dir = os.path.join(appdata, "Talk-to-Translate")
            else:
                log_dir = os.path.join(os.path.expanduser("~"), ".talk-to-translate")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "talk-to-translate.log")
            log_file = open(log_path, "a", encoding="utf-8", errors="replace")
        except Exception:
            log_file = None

    if sys.stdout is None:
        sys.stdout = _SafeStream(log_file)
    if sys.stderr is None:
        sys.stderr = _SafeStream(log_file)
    if sys.stdin is None:
        sys.stdin = _SafeReader()


# エントリーポイントの最上段で初期化を実行
_setup_safe_environment()
multiprocessing.freeze_support()

from src.ui import AppUI


def main():
    app = AppUI()
    app.mainloop()


if __name__ == "__main__":
    main()

