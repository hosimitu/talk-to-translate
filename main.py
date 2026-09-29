"""
Talk-to-Translate アプリケーション エントリーポイント
"""

from src.ui import AppUI


def main():
    app = AppUI()
    app.mainloop()


if __name__ == "__main__":
    main()
