"""
Talk-to-Translate 翻訳モジュール
クラウド翻訳（Google翻訳 / MyMemory）および完全ローカルAI翻訳（CTranslate2 + NLLB-200）に対応。
サーキットブレーカーによる自動切替とオフラインローカル翻訳のシームレスな切替を提供します。
"""

import os
import re
import time
from typing import Callable, Optional
from deep_translator import GoogleTranslator, MyMemoryTranslator


def parse_target_language_code(ui_language_text: str) -> str:
    """UIの翻訳先言語選択肢から言語コードへ変換"""
    if "ja" in ui_language_text:
        return "ja"
    elif "zh" in ui_language_text:
        return "zh-CN"
    return "en"


def parse_source_language_code(ui_language_text: str) -> str:
    """UIの入力言語選択肢から言語コードへ変換"""
    if "ja" in ui_language_text:
        return "ja"
    elif "zh" in ui_language_text:
        return "zh-CN"
    elif "en" in ui_language_text:
        return "en"
    return "auto"


def detect_language(text: str) -> str:
    """テキストから言語を簡易自動判定（ja, zh-CN, en）"""
    stripped = text.strip()
    if not stripped:
        return "en"
    # ひらがな・カタカナが含まれている場合は日本語
    if re.search(r"[\u3040-\u309F\u30A0-\u30FF]", stripped):
        return "ja"
    # 漢字が含まれており、ひらがな・カタカナがない場合は中国語
    if re.search(r"[\u4E00-\u9FFF]", stripped):
        return "zh-CN"
    # その他は英語とみなす
    return "en"


class LocalNLLBTranslator:
    """CTranslate2 と NLLB-200 (int8量子化) を用いたローカルAI翻訳エンジン"""

    MODELS = {
        "nllb-600M": "JustFrederik/nllb-200-distilled-600M-ct2-int8",
        "nllb-1.3B": "JustFrederik/nllb-200-1.3B-ct2-int8",
        "nllb-3.3B": "OpenNMT/nllb-200-3.3B-ct2-int8",
    }

    NLLB_LANG_CODES = {
        "ja": "jpn_Jpan",
        "en": "eng_Latn",
        "zh-CN": "zho_Hans",
        "zh": "zho_Hans",
    }

    def __init__(self, model_name: str = "nllb-600M", checkpoints_dir: Optional[str] = None):
        self.model_name = model_name
        self.repo_id = self.MODELS.get(model_name, self.MODELS["nllb-600M"])
        default_checkpoint = os.path.join("checkpoints", self.repo_id.split("/")[-1])
        self.checkpoints_dir = checkpoints_dir or default_checkpoint
        self.translator = None
        self.tokenizer = None
        self._is_loaded = False

    def load_model(self, progress_callback: Optional[Callable[[str], None]] = None):
        """モデルをダウンロード＆CTranslate2 Translatorとトークナイザーをロード"""
        if self._is_loaded:
            return

        if progress_callback:
            progress_callback(f"ローカル翻訳モデル ({self.model_name}) の準備・ロード中...")

        # Hugging Face Hubから必要ファイルをダウンロード（キャッシュがあればスキップ）
        from huggingface_hub import snapshot_download
        import ctranslate2
        from transformers import AutoTokenizer, utils as hf_utils

        # TransformersのMistral用正規表現誤検知警告を抑制
        hf_utils.logging.set_verbosity_error()

        model_dir = snapshot_download(repo_id=self.repo_id, local_dir=self.checkpoints_dir)

        # GPUが使える場合はCUDA、それ以外はCPUでint8高速推論
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        self.translator = ctranslate2.Translator(model_dir, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._is_loaded = True

        if progress_callback:
            progress_callback(f"ローカル翻訳モデル準備完了 ({device})")

    def translate(self, text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
        """
        ローカルモデルでテキストを翻訳する
        :param text: 翻訳元テキスト
        :param source_lang: 入力言語コード ("auto", "ja", "en", "zh-CN")
        :param target_lang: 出力言語コード ("ja", "en", "zh-CN")
        :return: 翻訳結果
        """
        stripped_text = text.strip()
        if not stripped_text:
            return ""

        if not self._is_loaded:
            self.load_model()

        if source_lang == "auto":
            source_lang = detect_language(stripped_text)

        src_nllb = self.NLLB_LANG_CODES.get(source_lang, "eng_Latn")
        tgt_nllb = self.NLLB_LANG_CODES.get(target_lang, "eng_Latn")

        # ソースとターゲットが同じならそのまま返す
        if src_nllb == tgt_nllb:
            return stripped_text

        try:
            self.tokenizer.src_lang = src_nllb
            source_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(stripped_text))
            results = self.translator.translate_batch(
                [source_tokens],
                target_prefix=[[tgt_nllb]],
                beam_size=5,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
            target_tokens = results[0].hypotheses[0][1:]
            translated = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(target_tokens)).strip()
            return translated
        except Exception as e:
            print(f"[LocalNLLBTranslator] ローカル翻訳エラー: {e}")
            raise


class TextTranslator:
    """サーキットブレーカー付きクラウド翻訳およびローカルAI翻訳の切替エンジン"""

    def __init__(
        self,
        source_lang: str = "auto",
        cooldown_sec: float = 300.0,
        engine: str = "cloud",
        nllb_model: str = "nllb-600M",
    ):
        """
        初期化
        :param source_lang: ソース言語 (デフォルト: "auto" で自動判別)
        :param cooldown_sec: Google翻訳エラー時の待機時間（秒、デフォルト: 5分）
        :param engine: 翻訳エンジン ("cloud" または "local")
        """
        self.source_lang = source_lang
        self.cooldown_sec = cooldown_sec
        self.engine = engine  # "cloud" または "local"
        self.nllb_model = nllb_model

        # Google翻訳の再開予定時刻（0なら使用可能）
        self._google_disabled_until = 0.0

        # ローカル翻訳エンジン（初期化は遅延）
        self.local_translator: Optional[LocalNLLBTranslator] = None

    def set_nllb_model(self, nllb_model: str):
        """ローカル翻訳モデルを変更する"""
        if self.nllb_model != nllb_model:
            self.nllb_model = nllb_model
            self.local_translator = None  # 強制リロード

    def set_engine(self, engine: str):
        """翻訳エンジンを切り替える ("cloud" または "local")"""
        self.engine = engine

    def ensure_local_loaded(self, progress_callback: Optional[Callable[[str], None]] = None):
        """ローカル翻訳モデルが未ロードなら事前ロードする"""
        if self.local_translator is None:
            self.local_translator = LocalNLLBTranslator(model_name=self.nllb_model)
        self.local_translator.load_model(progress_callback=progress_callback)

    def _get_mymemory_lang(self, lang: str) -> str:
        """MyMemory用の言語コード変換"""
        mapping = {
            "ja": "ja-JP",
            "en": "en-US",
            "zh-CN": "zh-CN",
            "auto": "ja-JP",
        }
        return mapping.get(lang, lang)

    def _translate_cloud(self, text: str, target_lang: str, source_lang: str) -> str:
        """クラウド翻訳（Google翻訳 ➔ レート制限時 MyMemory翻訳）"""
        now = time.time()

        # 1. Google翻訳の試行（サーキットブレーカーが開いていない場合）
        if now >= self._google_disabled_until:
            try:
                translator = GoogleTranslator(source=source_lang, target=target_lang)
                translated = translator.translate(text)
                if translated:
                    return translated
            except Exception as e_google:
                # レート制限等のエラー検知時：サーキットブレーカーを発動（クールダウン開始）
                self._google_disabled_until = now + self.cooldown_sec
                cooldown_min = int(self.cooldown_sec // 60)
                print(f"[TextTranslator] Google翻訳一時制限を検知しました。今後{cooldown_min}分間は直接MyMemory翻訳を使用します。")

        # 2. MyMemoryTranslator による高速翻訳
        try:
            if source_lang == "auto":
                detected = detect_language(text)
                s_lang = self._get_mymemory_lang(detected)
            else:
                s_lang = self._get_mymemory_lang(source_lang)
            t_lang = self._get_mymemory_lang(target_lang)
            if s_lang == t_lang:
                return text

            mm_translator = MyMemoryTranslator(source=s_lang, target=t_lang)
            translated = mm_translator.translate(text)
            if translated:
                return translated
        except Exception as e_mm:
            print(f"[TextTranslator] MyMemory翻訳エラー: {e_mm}")

        return f"[翻訳不可: {text}]"

    def translate(
        self,
        text: str,
        target_lang: str = "en",
        source_lang: Optional[str] = None,
    ) -> str:
        """
        テキストを指定の言語へ翻訳する
        :param text: 翻訳元のテキスト
        :param target_lang: 翻訳先言語コード ("en", "ja", "zh-CN")
        :param source_lang: 入力言語コード ("auto", "ja", "en", "zh-CN" など)
        :return: 翻訳されたテキスト
        """
        stripped_text = text.strip()
        if not stripped_text:
            return ""

        speaker_prefix = ""
        spk_match = re.match(r"^(\[話者\d+\])\s*(.*)$", stripped_text)
        if spk_match:
            speaker_prefix = spk_match.group(1) + " "
            stripped_text = spk_match.group(2).strip()
            if not stripped_text:
                return speaker_prefix.strip()

        effective_source_lang = source_lang or self.source_lang

        # ローカルAI翻訳モード
        if self.engine == "local":
            try:
                if self.local_translator is None:
                    self.local_translator = LocalNLLBTranslator(model_name=self.nllb_model)
                res = self.local_translator.translate(
                    stripped_text,
                    source_lang=effective_source_lang,
                    target_lang=target_lang,
                )
                return f"{speaker_prefix}{res}"
            except Exception as e_local:
                print(f"[TextTranslator] ローカル翻訳失敗のためクラウドへフォールバックします: {e_local}")
                # クラウド翻訳へフォールバック

        # クラウド翻訳モード
        cloud_res = self._translate_cloud(stripped_text, target_lang=target_lang, source_lang=effective_source_lang)
        return f"{speaker_prefix}{cloud_res}"
