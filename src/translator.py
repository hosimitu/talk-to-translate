"""
Talk-to-Translate 翻訳モジュール
deep-translator を使用してリアルタイムに多言語テキストを翻訳します。
GoogleTranslator および MyMemoryTranslator への自動フォールバックを備えています。
"""

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
    """Google翻訳（プライマリ）とMyMemory（フォールバック）を統合した堅牢な翻訳エンジン"""

    def __init__(self, source_lang: str = "auto"):
        """
        初期化
        :param source_lang: ソース言語 (デフォルト: "auto" で自動判別)
        """
        self.source_lang = source_lang

    def _get_mymemory_lang(self, lang: str) -> str:
        """MyMemory用の言語コード変換"""
        mapping = {
            "ja": "ja-JP",
            "en": "en-US",
            "zh-CN": "zh-CN",
            "auto": "ja-JP",  # auto不可時のデフォルト
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

        # 1. GoogleTranslator を試行
        try:
            translator = GoogleTranslator(source=self.source_lang, target=target_lang)
            translated = translator.translate(stripped_text)
            if translated:
                return translated
        except Exception as e_google:
            print(f"[TextTranslator] Google翻訳一時エラー、MyMemoryへフォールバックします: {e_google}")

        # 2. MyMemoryTranslator へフォールバック
        try:
            s_lang = self._get_mymemory_lang(self.source_lang)
            t_lang = self._get_mymemory_lang(target_lang)
            # ソースと言語が同じ場合は翻訳不要
            if s_lang == t_lang:
                return stripped_text
            mm_translator = MyMemoryTranslator(source=s_lang, target=t_lang)
            translated = mm_translator.translate(stripped_text)
            if translated:
                return translated
        except Exception as e_mm:
            print(f"[TextTranslator] フォールバック翻訳も失敗: {e_mm}")

        return f"[翻訳不可: {stripped_text}]"
