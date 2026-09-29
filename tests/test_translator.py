"""翻訳エンジンの単体テストスクリプト"""

from src.translator import parse_target_language_code, TextTranslator


def test_parsers():
    assert parse_target_language_code("英語 (en)") == "en"
    assert parse_target_language_code("日本語 (ja)") == "ja"
    assert parse_target_language_code("中国語 (zh)") == "zh-CN"
    print("[OK] 翻訳言語パース関数のテスト成功")


def test_translation():
    print("翻訳機能（フォールバック付き）のテスト実行中...")
    translator = TextTranslator(source_lang="auto")

    # 日本語 -> 英語
    res_en = translator.translate("こんにちは", target_lang="en")
    print(f"[OK] 翻訳結果 (こんにちは -> en): '{res_en}'")
    assert len(res_en) > 0
    assert not res_en.startswith("[翻訳不可")


if __name__ == "__main__":
    test_parsers()
    test_translation()
    print("すべての翻訳テストが正常に通過しました！")
