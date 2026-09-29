"""
Talk-to-Translate 翻訳モジュール
deep-translator を使用してリアルタイムに多言語テキストを翻訳します。
サーキットブレーカー付きのGoogle翻訳およびMyMemory翻訳の自動切替を備えています。
"""

import time
from typing import Optional
from deep_translator import GoogleTranslator, MyMemoryTranslator


def parse_target_language_code(ui_language_text: str) -> str:
    """UIの翻訳先言語選択肢から言語コードへ変換"""
    if "ja" in ui_language_text:
        return "ja"
    elif "zh" in ui_language_text:
        return "zh-CN"
    return "en"


class TextTranslator:
    """サーキットブレーカー付きの堅牢な翻訳エンジン"""

    def __init__(self, source_lang: str = "auto", cooldown_sec: float = 300.0):
        """
        初期化
        :param source_lang: ソース言語 (デフォルト: "auto" で自動判別)
        :param cooldown_sec: Google翻訳エラー時の待機時間（秒、デフォルト: 5分）
        """
        self.source_lang = source_lang
        self.cooldown_sec = cooldown_sec
        # Google翻訳の再開予定時刻（0なら使用可能）
        self._google_disabled_until = 0.0

    def _get_mymemory_lang(self, lang: str) -> str:
        """MyMemory用の言語コード変換"""
        mapping = {
            "ja": "ja-JP",
            "en": "en-US",
            "zh-CN": "zh-CN",
            "auto": "ja-JP",
        }
        return mapping.get(lang, lang)

    def translate(self, text: str, target_lang: str = "en") -> str:
        """
        テキストを指定の言語へ翻訳する
        :param text: 翻訳元のテキスト
        :param target_lang: 翻訳先言語コード ("en", "ja", "zh-CN")
        :return: 翻訳されたテキスト
        """
        stripped_text = text.strip()
        if not stripped_text:
            return ""

        now = time.time()

        # 1. Google翻訳の試行（サーキットブレーカーが開いていない場合）
        if now >= self._google_disabled_until:
            try:
                translator = GoogleTranslator(source=self.source_lang, target=target_lang)
                translated = translator.translate(stripped_text)
                if translated:
                    return translated
            except Exception as e_google:
                # レート制限等のエラー検知時：サーキットブレーカーを発動（クールダウン開始）
                self._google_disabled_until = now + self.cooldown_sec
                cooldown_min = int(self.cooldown_sec // 60)
                print(f"[TextTranslator] Google翻訳一時制限を検知しました。今後{cooldown_min}分間は直接MyMemory翻訳を使用します。")

        # 2. MyMemoryTranslator による高速翻訳
        try:
            s_lang = self._get_mymemory_lang(self.source_lang)
            t_lang = self._get_mymemory_lang(target_lang)
            if s_lang == t_lang:
                return stripped_text

            mm_translator = MyMemoryTranslator(source=s_lang, target=t_lang)
            translated = mm_translator.translate(stripped_text)
            if translated:
                return translated
        except Exception as e_mm:
            print(f"[TextTranslator] 翻訳処理エラー: {e_mm}")

        return f"[翻訳不可: {stripped_text}]"
