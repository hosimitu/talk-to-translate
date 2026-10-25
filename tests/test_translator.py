"""翻訳エンジンの単体テストスクリプト"""

import time
from src.translator import (
    parse_target_language_code,
    parse_source_language_code,
    parse_translation_engine,
    canonical_language_code,
    detect_language,
    resolve_local_device,
    LocalNLLBTranslator,
    TextTranslator,
)


def test_parsers():
    # 翻訳先パース
    assert parse_target_language_code("英語 (en)") == "en"
    assert parse_target_language_code("日本語 (ja)") == "ja"
    assert parse_target_language_code("中国語 (zh)") == "zh-CN"

    # 入力言語パース
    assert parse_source_language_code("自動検出 (auto)") == "auto"
    assert parse_source_language_code("日本語 (ja)") == "ja"
    assert parse_source_language_code("英語 (en)") == "en"
    assert parse_source_language_code("中国語 (zh)") == "zh-CN"

    # 翻訳エンジン選択肢のパース（UIラベル -> 内部エンジン/モデル）
    assert parse_translation_engine("クラウド翻訳 (Google/MyMemory) (推奨)") == ("cloud", "nllb-600M")
    assert parse_translation_engine("ローカル翻訳 (NLLB 600M) (推奨)") == ("local", "nllb-600M")
    assert parse_translation_engine("ローカル翻訳 (NLLB 1.3B) (推奨)") == ("local", "nllb-1.3B")
    assert parse_translation_engine("ローカル翻訳 (NLLB 3.3B)") == ("local", "nllb-3.3B")
    # 旧表記
    assert parse_translation_engine("ローカル翻訳 (NLLB 600M)") == ("local", "nllb-600M")

    # 言語簡易自動判定
    assert detect_language("こんにちは世界") == "ja"
    assert detect_language("お元気ですか？") == "ja"
    assert detect_language("你好，世界") == "zh-CN"
    assert detect_language("Hello, how are you?") == "en"

    # 言語コードの表記ゆれ統一
    assert canonical_language_code("zh") == canonical_language_code("zh-CN") == "zh"
    assert canonical_language_code("JA") == "ja"
    assert canonical_language_code("") == ""
    print("[OK] 言語パースおよび簡易検出関数のテスト成功")


def test_circuit_breaker():
    print("サーキットブレーカー付きクラウド翻訳テスト実行中...")
    translator = TextTranslator(source_lang="auto", cooldown_sec=10.0, engine="cloud")

    # 1回目の翻訳（Googleがレート制限ならここでエラーを検知してブレーカー発動）
    start_t = time.time()
    res1 = translator.translate("こんにちは", target_lang="en")
    dur1 = time.time() - start_t
    print(f"[OK] 1回目翻訳 (Cloud): '{res1}' (所要時間: {dur1:.2f}秒)")
    assert len(res1) > 0

    # 2回目の翻訳（ブレーカー発動中ならMyMemoryへ直接行く）
    start_t2 = time.time()
    res2 = translator.translate("ありがとう", target_lang="en")
    dur2 = time.time() - start_t2
    print(f"[OK] 2回目翻訳 (Cloud): '{res2}' (所要時間: {dur2:.2f}秒)")
    assert len(res2) > 0


def test_local_nllb_translator():
    print("ローカルAI翻訳 (NLLB-200) テスト実行中...")
    local_trans = LocalNLLBTranslator()
    local_trans.load_model()

    # 1. 英語 -> 日本語
    res_en_ja = local_trans.translate("Good morning!", source_lang="en", target_lang="ja")
    print(f"[OK] ローカル翻訳 EN -> JA: '{res_en_ja}'")
    assert len(res_en_ja) > 0

    # 2. 中国語 -> 日本語
    res_zh_ja = local_trans.translate("谢谢你的帮助。", source_lang="zh-CN", target_lang="ja")
    print(f"[OK] ローカル翻訳 ZH -> JA: '{res_zh_ja}'")
    assert len(res_zh_ja) > 0

    # 3. 日本語 -> 英語
    res_ja_en = local_trans.translate("よろしくお願いします。", source_lang="ja", target_lang="en")
    print(f"[OK] ローカル翻訳 JA -> EN: '{res_ja_en}'")
    assert len(res_ja_en) > 0


def test_translator_engine_switching():
    print("翻訳エンジン切替テスト実行中...")
    translator = TextTranslator(source_lang="auto", engine="cloud")
    assert translator.engine == "cloud"

    # ローカルAIへの切替
    translator.set_engine("local")
    assert translator.engine == "local"
    res_local = translator.translate("Hello world", target_lang="ja", source_lang="en")
    print(f"[OK] エンジン切替後 (Local): '{res_local}'")
    assert len(res_local) > 0

    # クラウドへの切替戻し
    translator.set_engine("cloud")
    assert translator.engine == "cloud"
    res_cloud = translator.translate("Hello world", target_lang="ja", source_lang="en")
    print(f"[OK] エンジン切替後 (Cloud): '{res_cloud}'")
    assert len(res_cloud) > 0


def test_cache_and_same_language_skip():
    """同一言語スキップと翻訳キャッシュ（ネットワーク非依存）"""
    calls = []
    translator = TextTranslator(engine="cloud")

    def fake_cloud(text, target_lang, source_lang):
        calls.append((text, target_lang, source_lang))
        if text == "失敗":
            return "[翻訳不可: 失敗]"
        return f"<{text}>"

    translator._translate_cloud = fake_cloud

    # 1. 同一言語はクラウドを呼ばず即返す
    assert translator.translate("こんにちは", target_lang="ja", source_lang="ja") == "こんにちは"
    assert translator.translate("你好", target_lang="zh-CN", source_lang="zh") == "你好"
    assert calls == []

    # 2. 同じテキストは2回目以降キャッシュから返る
    assert translator.translate("hello", target_lang="ja", source_lang="en") == "<hello>"
    assert translator.translate("hello", target_lang="ja", source_lang="en") == "<hello>"
    assert len(calls) == 1

    # 3. 話者タグはキャッシュ利用時も保持される
    assert translator.translate("[話者1] good morning", target_lang="ja", source_lang="en") == "[話者1] <good morning>"
    assert translator.translate("[話者1] good morning", target_lang="ja", source_lang="en") == "[話者1] <good morning>"
    assert len(calls) == 2

    # 4. モデル変更でキャッシュが破棄される
    translator.set_nllb_model("nllb-1.3B")
    assert translator.translate("hello", target_lang="ja", source_lang="en") == "<hello>"
    assert len(calls) == 3

    # 5. 失敗結果（[翻訳不可]）はキャッシュしない
    assert translator.translate("失敗", target_lang="ja", source_lang="en").startswith("[翻訳不可")
    assert translator.translate("失敗", target_lang="ja", source_lang="en").startswith("[翻訳不可")
    assert len(calls) == 5
    print("[OK] 同一言語スキップ・翻訳キャッシュのテスト成功")


def test_local_device_resolution():
    """ローカル翻訳のデバイス判定が cpu / cuda のいずれかを返すこと"""
    device = resolve_local_device()
    assert device in ("cpu", "cuda")
    print(f"[OK] ローカル翻訳デバイス判定テスト成功 (device={device})")


if __name__ == "__main__":
    test_parsers()
    test_cache_and_same_language_skip()
    test_local_device_resolution()
    test_circuit_breaker()
    test_local_nllb_translator()
    test_translator_engine_switching()
    print("すべての翻訳テストが正常に通過しました！")
