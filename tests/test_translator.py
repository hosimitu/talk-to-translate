"""翻訳エンジンの単体テストスクリプト"""

import time
from src.translator import parse_target_language_code, TextTranslator


def test_parsers():
    assert parse_target_language_code("英語 (en)") == "en"
    assert parse_target_language_code("日本語 (ja)") == "ja"
    assert parse_target_language_code("中国語 (zh)") == "zh-CN"
    print("[OK] 翻訳言語パース関数のテスト成功")


def test_circuit_breaker():
    print("サーキットブレーカー付き翻訳テスト実行中...")
    translator = TextTranslator(source_lang="auto", cooldown_sec=10.0)

    # 1回目の翻訳（Googleがレート制限ならここでエラーを検知してブレーカー発動）
    start_t = time.time()
    res1 = translator.translate("こんにちは", target_lang="en")
    dur1 = time.time() - start_t
    print(f"[OK] 1回目翻訳: '{res1}' (所要時間: {dur1:.2f}秒)")
    assert len(res1) > 0

    # 2回目の翻訳（ブレーカー発動中なのでGoogleをスキップし即座にMyMemoryへ行く）
    start_t2 = time.time()
    res2 = translator.translate("ありがとう", target_lang="en")
    dur2 = time.time() - start_t2
    print(f"[OK] 2回目翻訳: '{res2}' (所要時間: {dur2:.2f}秒)")
    assert len(res2) > 0


if __name__ == "__main__":
    test_parsers()
    test_circuit_breaker()
    print("すべての翻訳テストが正常に通過しました！")
