"""
標準ストリーム保護機能の単体テスト
PyInstallerのwindowedモード (console=False) 下で sys.stdout/stderr/stdin が None の場合に
安全なストリームへ置き換わることを検証します。
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import main


def test_safe_environment_with_none_streams():
    """sys.stdout / sys.stderr / sys.stdin が None の場合の安全化テスト"""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_stdin = sys.stdin

    try:
        # Windowed環境をシミュレート
        sys.stdout = None
        sys.stderr = None
        sys.stdin = None

        main._setup_safe_environment()

        # NoneType ではなく安全なストリームに置き換わっているか確認
        assert sys.stdout is not None
        assert sys.stderr is not None
        assert sys.stdin is not None

        # write, flush, isatty が例外なく呼べるか確認 (AttributeError が出ないこと)
        sys.stdout.write("stdout test\n")
        sys.stdout.flush()
        assert not sys.stdout.isatty()

        sys.stderr.write("stderr test\n")
        sys.stderr.flush()
        assert not sys.stderr.isatty()

        assert sys.stdin.read() == ""
        assert sys.stdin.readline() == ""

        # 環境変数の設定確認
        assert os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") == "1"
        assert os.environ.get("MODELSCOPE_DISABLE_PROGRESS_BAR") == "1"

        print("[OK] 標準ストリーム安全化テストに成功しました。")

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        sys.stdin = old_stdin


if __name__ == "__main__":
    test_safe_environment_with_none_streams()
    print("すべてのストリーム保護テストが正常に通過しました！")
