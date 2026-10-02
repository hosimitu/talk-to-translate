import queue
import re
import threading
import tkinter as tk
from typing import Optional
import customtkinter as ctk
import sounddevice as sd
from src.config import ConfigManager
from src.audio import AudioRecorder
from src.transcriber import TranscriptionEngine, parse_language_code, parse_model_size
from src.processor import AudioProcessor
from src.diarizer import SpeakerDiarizer
from src.translator import (
    TextTranslator,
    parse_target_language_code,
    parse_source_language_code,
)
from src.separator import SpeechSeparator

# テーマ設定
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


def get_preferred_font(size: int = 16, weight: str = "normal") -> ctk.CTkFont:
    """環境に応じた視認性の高いフォント（Windows: Yu Gothic UI / Meiryo / BIZ UDPGothic など）を取得"""
    candidates = ["Yu Gothic UI", "BIZ UDPGothic", "Meiryo", "Hiragino Sans", "Noto Sans CJK JP"]
    try:
        import tkinter.font as tkfont
        available = set(tkfont.families())
        for fam in candidates:
            if fam in available:
                return ctk.CTkFont(family=fam, size=size, weight=weight)
    except Exception:
        pass
    return ctk.CTkFont(family="Yu Gothic UI", size=size, weight=weight)


class AppUI(ctk.CTk):
    """メインアプリケーションウィンドウ"""

    def __init__(self):
        super().__init__()

        self.title("Talk-to-Translate")
        self.geometry("900x720")
        self.minsize(800, 620)

        # 設定マネージャーの初期化と設定読み込み
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()

        # 音声レコーダーの初期化
        self.recorder = AudioRecorder(sample_rate=16000)

        # 文字起こしエンジン & プロセッサ
        self.transcriber: TranscriptionEngine | None = None
        self.processor: AudioProcessor | None = None

        # 話者識別エンジン & 音源分離エンジン
        self.diarizer = SpeakerDiarizer()
        self.separator = SpeechSeparator()

        # 翻訳エンジンの初期化（保存された設定を反映）
        saved_engine = self.config.get("translation_engine", "")
        is_local = "ローカル" in saved_engine
        nllb_model = "nllb-600M"
        if "1.3B" in saved_engine:
            nllb_model = "nllb-1.3B"
        elif "3.3B" in saved_engine:
            nllb_model = "nllb-3.3B"

        self.translator = TextTranslator(
            source_lang="auto",
            engine="local" if is_local else "cloud",
            nllb_model=nllb_model,
        )

        # スレッド間通信用キュー
        self.transcription_queue: queue.Queue[str] = queue.Queue()
        self.translation_queue: queue.Queue[str] = queue.Queue()

        # 録音中フラグ
        self.is_recording = False

        # 翻訳用小バッファリング管理 (2〜3行 / 約70文字 / 2.0秒休止フラッシュ)
        self._translation_buffer: list[str] = []
        self._buffer_lock = threading.Lock()
        self._buffer_timer: Optional[threading.Timer] = None
        self._buffer_flush_delay = 2.0
        self._buffer_max_lines = 3
        self._buffer_max_chars = 70

        # グリッドのウェイト設定（中央のテキストエリアが伸縮）
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # UIコンポーネントの構築
        self._build_header_panel()
        self._build_text_areas()
        self._build_footer_panel()

        # ウィンドウ終了時のハンドラ登録
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        # キューの監視ループを開始
        self._poll_queues()

    def _poll_queues(self):
        """ワーカースレッドからのキューをメインスレッドで安全に処理"""
        # 文字起こしキューの処理
        while not self.transcription_queue.empty():
            try:
                text = self.transcription_queue.get_nowait()
                self.append_transcription(text)
            except queue.Empty:
                break

        # 翻訳キューの処理
        while not self.translation_queue.empty():
            try:
                translated_text = self.translation_queue.get_nowait()
                self.append_translation(translated_text)
            except queue.Empty:
                break

        # 50ミリ秒ごとに再帰チェック
        self.after(50, self._poll_queues)

    def _get_audio_input_devices(self) -> list[str]:
        """利用可能なマイク（入力）デバイスの一覧を取得"""
        devices = []
        try:
            device_list = sd.query_devices()
            for i, dev in enumerate(device_list):
                if dev.get("max_input_channels", 0) > 0:
                    devices.append(f"{i}: {dev['name']}")
        except Exception:
            devices = ["0: デフォルトマイク"]
        return devices if devices else ["0: デフォルトマイク"]

    def _build_header_panel(self):
        """上部の設定・操作パネル"""
        header_frame = ctk.CTkFrame(self, corner_radius=10)
        header_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        # 4列均等配置
        header_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # 1. マイク選択
        mic_label = ctk.CTkLabel(header_frame, text="マイクデバイス:", font=ctk.CTkFont(size=12, weight="bold"))
        mic_label.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")
        
        mics = self._get_audio_input_devices()
        self.mic_option = ctk.CTkOptionMenu(
            header_frame,
            values=mics,
            command=self._save_current_settings,
        )
        self.mic_option.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")

        # 保存されていたマイクデバイスの復元
        saved_mic = self.config.get("mic_device", "")
        matched_mic = next((m for m in mics if saved_mic and saved_mic in m), None)
        if matched_mic:
            self.mic_option.set(matched_mic)
        elif mics:
            self.mic_option.set(mics[0])

        # 2. 文字起こしモデル選択
        model_label = ctk.CTkLabel(header_frame, text="AIモデル:", font=ctk.CTkFont(size=12, weight="bold"))
        model_label.grid(row=0, column=1, padx=10, pady=(10, 0), sticky="w")
        
        models = [
            "SenseVoice-Small (超高速・日中英推奨)",
            "Whisper: large-v3-turbo (高精度)",
            "Whisper: small (軽量)",
        ]
        self.model_option = ctk.CTkOptionMenu(
            header_frame,
            values=models,
            command=self._save_current_settings,
        )
        self.model_option.grid(row=1, column=1, padx=10, pady=(0, 10), sticky="ew")

        # 保存されていた文字起こしモデルの復元
        saved_model = self.config.get("transcription_model")
        if saved_model in models:
            self.model_option.set(saved_model)
        else:
            self.model_option.set("SenseVoice-Small (超高速・日中英推奨)")

        # 3. 入力音声言語
        lang_label = ctk.CTkLabel(header_frame, text="入力言語 (音声):", font=ctk.CTkFont(size=12, weight="bold"))
        lang_label.grid(row=0, column=2, padx=10, pady=(10, 0), sticky="w")
        
        languages = ["自動検出 (auto)", "日本語 (ja)", "英語 (en)", "中国語 (zh)"]
        self.input_lang_option = ctk.CTkOptionMenu(
            header_frame,
            values=languages,
            command=self._save_current_settings,
        )
        self.input_lang_option.grid(row=1, column=2, padx=10, pady=(0, 10), sticky="ew")

        # 保存されていた入力言語の復元
        saved_input_lang = self.config.get("input_lang")
        if saved_input_lang in languages:
            self.input_lang_option.set(saved_input_lang)
        else:
            self.input_lang_option.set("自動検出 (auto)")

        # 4. 翻訳先言語
        trans_label = ctk.CTkLabel(header_frame, text="翻訳先言語:", font=ctk.CTkFont(size=12, weight="bold"))
        trans_label.grid(row=0, column=3, padx=10, pady=(10, 0), sticky="w")
        
        target_languages = ["英語 (en)", "日本語 (ja)", "中国語 (zh)"]
        self.target_lang_option = ctk.CTkOptionMenu(
            header_frame,
            values=target_languages,
            command=self._save_current_settings,
        )
        self.target_lang_option.grid(row=1, column=3, padx=10, pady=(0, 10), sticky="ew")

        # 保存されていた翻訳先言語の復元
        saved_target_lang = self.config.get("target_lang")
        if saved_target_lang in target_languages:
            self.target_lang_option.set(saved_target_lang)
        else:
            self.target_lang_option.set("英語 (en)")

        # 5. 翻訳エンジン選択 (2行目の前半)
        engine_label = ctk.CTkLabel(header_frame, text="翻訳エンジン:", font=ctk.CTkFont(size=12, weight="bold"))
        engine_label.grid(row=2, column=0, padx=10, pady=(6, 12), sticky="w")

        engines = [
            "クラウド翻訳 (Google/MyMemory) (推奨)",
            "ローカル翻訳 (NLLB 600M) (推奨)",
            "ローカル翻訳 (NLLB 1.3B) (推奨)",
            "ローカル翻訳 (NLLB 3.3B)",
        ]
        self.engine_option = ctk.CTkOptionMenu(
            header_frame,
            values=engines,
            command=self._on_change_engine,
        )
        self.engine_option.grid(row=2, column=1, padx=10, pady=(6, 12), sticky="ew")

        # 保存されていた翻訳エンジンの復元
        saved_engine = self.config.get("translation_engine")
        if saved_engine in engines:
            self.engine_option.set(saved_engine)
        elif saved_engine and "600M" in saved_engine:
            self.engine_option.set("ローカル翻訳 (NLLB 600M) (推奨)")
        elif saved_engine and "1.3B" in saved_engine:
            self.engine_option.set("ローカル翻訳 (NLLB 1.3B) (推奨)")
        elif saved_engine and "3.3B" in saved_engine:
            self.engine_option.set("ローカル翻訳 (NLLB 3.3B)")
        else:
            self.engine_option.set("クラウド翻訳 (Google/MyMemory) (推奨)")

        # 6. 話者識別（旧・音源分離）チェックボックス (2行目の後半に配置)
        self.sep_checkbox = ctk.CTkCheckBox(
            header_frame,
            text="👥 話者識別（3D-speaker ONNX / 多人数対応）",
            font=ctk.CTkFont(size=12),
            command=self._on_toggle_separation,
        )
        self.sep_checkbox.grid(row=2, column=2, columnspan=2, padx=15, pady=(6, 12), sticky="w")

        # 保存されていた音源分離の復元
        if self.config.get("enable_separation", False):
            self.sep_checkbox.select()
        else:
            self.sep_checkbox.deselect()

    def _build_text_areas(self):
        """中央のテキストエリア（文字起こし結果と翻訳結果の2ペイン）"""
        content_frame = ctk.CTkFrame(self, fg_color="transparent")
        content_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure((0, 1), weight=1)

        # --- 左側: 文字起こし結果 ---
        left_frame = ctk.CTkFrame(content_frame, corner_radius=10)
        left_frame.grid(row=0, column=0, padx=(0, 10), pady=0, sticky="nsew")
        left_frame.grid_rowconfigure(1, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        # テキストエリア用フォント（視認性向上のため16pt & UDフォント）
        textbox_font = get_preferred_font(size=16)

        # 文字起こしヘッダー (ラベル & コピーボタン)
        transcribe_header = ctk.CTkFrame(left_frame, fg_color="transparent")
        transcribe_header.grid(row=0, column=0, padx=15, pady=(10, 5), sticky="ew")
        transcribe_header.grid_columnconfigure(0, weight=1)

        transcribe_title = ctk.CTkLabel(
            transcribe_header, 
            text="🎙️ 文字起こし結果", 
            font=get_preferred_font(size=15, weight="bold")
        )
        transcribe_title.grid(row=0, column=0, sticky="w")

        self.copy_transcribe_btn = ctk.CTkButton(
            transcribe_header, 
            text="コピー", 
            width=70, 
            height=28,
            command=self._copy_transcription
        )
        self.copy_transcribe_btn.grid(row=0, column=1, sticky="e")

        # 文字起こしテキストボックス（文字サイズ拡大・行間調整・視認性の高い配色）
        self.transcribe_textbox = ctk.CTkTextbox(
            left_frame,
            wrap="word",
            font=textbox_font,
            text_color=("#1a1a1a", "#f8f9fa"),
        )
        self.transcribe_textbox.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="nsew")
        self.transcribe_textbox._textbox.configure(spacing1=2, spacing2=4, spacing3=4)

        # --- 右側: 翻訳結果 ---
        right_frame = ctk.CTkFrame(content_frame, corner_radius=10)
        right_frame.grid(row=0, column=1, padx=(10, 0), pady=0, sticky="nsew")
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        # 翻訳ヘッダー (ラベル & コピーボタン)
        translate_header = ctk.CTkFrame(right_frame, fg_color="transparent")
        translate_header.grid(row=0, column=0, padx=15, pady=(10, 5), sticky="ew")
        translate_header.grid_columnconfigure(0, weight=1)

        translate_title = ctk.CTkLabel(
            translate_header, 
            text="🌐 翻訳結果", 
            font=get_preferred_font(size=15, weight="bold")
        )
        translate_title.grid(row=0, column=0, sticky="w")

        self.copy_translate_btn = ctk.CTkButton(
            translate_header, 
            text="コピー", 
            width=70, 
            height=28,
            command=self._copy_translation
        )
        self.copy_translate_btn.grid(row=0, column=1, sticky="e")

        # 翻訳テキストボックス（文字サイズ拡大・行間調整・視認性の高い配色）
        self.translate_textbox = ctk.CTkTextbox(
            right_frame,
            wrap="word",
            font=textbox_font,
            text_color=("#1a1a1a", "#f8f9fa"),
        )
        self.translate_textbox.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="nsew")
        self.translate_textbox._textbox.configure(spacing1=2, spacing2=4, spacing3=4)

    def _build_footer_panel(self):
        """下部のコントロール・ステータスパネル"""
        footer_frame = ctk.CTkFrame(self, corner_radius=10)
        footer_frame.grid(row=2, column=0, padx=20, pady=(10, 20), sticky="ew")
        footer_frame.grid_columnconfigure(0, weight=1)
        footer_frame.grid_columnconfigure(1, weight=2)
        footer_frame.grid_columnconfigure(2, weight=1)

        # ステータスラベル
        self.status_label = ctk.CTkLabel(
            footer_frame, 
            text="ステータス: 待機中", 
            font=ctk.CTkFont(size=13),
            text_color="gray"
        )
        self.status_label.grid(row=0, column=0, padx=20, pady=15, sticky="w")

        # 録音開始/停止ボタン
        self.record_button = ctk.CTkButton(
            footer_frame, 
            text="録音開始", 
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#28a745",
            hover_color="#218838",
            height=40,
            command=self._toggle_recording
        )
        self.record_button.grid(row=0, column=1, padx=20, pady=12, sticky="ew")

        # クリアボタン
        self.clear_button = ctk.CTkButton(
            footer_frame, 
            text="テキスト全消去", 
            fg_color="#6c757d",
            hover_color="#5a6268",
            width=100, 
            height=32,
            command=self._clear_all_text
        )
        self.clear_button.grid(row=0, column=2, padx=20, pady=15, sticky="e")

    def _get_selected_device_index(self) -> int | None:
        """選択されているマイクのデバイス番号を取得"""
        selected = self.mic_option.get()
        try:
            # "0: デバイス名" の形式から数字部分を抽出
            device_id_str = selected.split(":")[0].strip()
            return int(device_id_id) if (device_id_id := device_id_str).isdigit() else None
        except Exception:
            return None

    def _on_transcription_received(self, text: str):
        """バックグラウンドスレッドからの文字起こし結果を受け取りキューへ追加＆小バッファリング翻訳実行"""
        # 左側の文字起こしテキストボックスへは即座に反映
        self.transcription_queue.put(text)

        stripped = text.strip()
        if not stripped:
            return

        with self._buffer_lock:
            # 話者タグ（[話者1]等）の確認
            spk_match = re.match(r"^(\[話者\d+\])", stripped)
            current_speaker = spk_match.group(1) if spk_match else None

            last_speaker = None
            if self._translation_buffer:
                last_spk_match = re.match(r"^(\[話者\d+\])", self._translation_buffer[0])
                last_speaker = last_spk_match.group(1) if last_spk_match else None

            # 話者が交代した場合は混在を防ぐため直前のバッファをフラッシュ
            if current_speaker != last_speaker and self._translation_buffer:
                self._flush_translation_buffer_locked()

            # バッファに追加
            self._translation_buffer.append(stripped)

            # バッファ内の合計文字数
            total_chars = sum(len(line) for line in self._translation_buffer)

            # 2〜3行（または文字数閾値）に達した場合は即座にフラッシュ
            if len(self._translation_buffer) >= self._buffer_max_lines or total_chars >= self._buffer_max_chars:
                self._flush_translation_buffer_locked()
            else:
                # 達していない場合は発話休止タイマー（2.0秒）を再設定
                if self._buffer_timer is not None:
                    self._buffer_timer.cancel()
                self._buffer_timer = threading.Timer(
                    self._buffer_flush_delay,
                    self._on_buffer_timeout,
                )
                self._buffer_timer.daemon = True
                self._buffer_timer.start()

    def _on_buffer_timeout(self):
        """発話休止タイマー満了時の自動フラッシュ"""
        with self._buffer_lock:
            self._flush_translation_buffer_locked()

    def _flush_translation_buffer_locked(self):
        """ロック保持下でのバッファフラッシュと翻訳スレッド起動"""
        if self._buffer_timer is not None:
            self._buffer_timer.cancel()
            self._buffer_timer = None

        if not self._translation_buffer:
            return

        buffered_lines = list(self._translation_buffer)
        self._translation_buffer.clear()

        source_lang = parse_source_language_code(self.input_lang_option.get())
        target_lang = parse_target_language_code(self.target_lang_option.get())
        threading.Thread(
            target=self._run_buffered_translation_thread,
            args=(buffered_lines, target_lang, source_lang),
            daemon=True,
        ).start()

    def _run_buffered_translation_thread(self, lines: list[str], target_lang: str, source_lang: str = "auto"):
        """バッファされた複数行を自然に結合して翻訳を実行"""
        if not lines:
            return

        first_speaker = ""
        m = re.match(r"^(\[話者\d+\])\s*(.*)$", lines[0])
        if m:
            first_speaker = m.group(1)

        cleaned_parts = []
        for line in lines:
            sm = re.match(r"^(\[話者\d+\])\s*(.*)$", line)
            content = sm.group(2) if sm else line
            content = content.strip()
            if content:
                cleaned_parts.append(content)

        if not cleaned_parts:
            return

        # CJK（日本語・中国語）の場合は読点や句点等を考慮して結合
        first_content = cleaned_parts[0]
        is_cjk = bool(re.search(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]", first_content))
        if is_cjk:
            combined_text = ""
            for i, part in enumerate(cleaned_parts):
                if i > 0 and combined_text and combined_text[-1] not in "。！？!?.,、， ":
                    combined_text += "。"
                combined_text += part
        else:
            combined_text = " ".join(cleaned_parts)

        if first_speaker:
            combined_text = f"{first_speaker} {combined_text}"

        self._run_translation_thread(combined_text, target_lang, source_lang)

    def _run_translation_thread(self, text: str, target_lang: str, source_lang: str = "auto"):
        """別スレッドで翻訳を実行し、翻訳キューへ追加"""
        translated_text = self.translator.translate(
            text, target_lang=target_lang, source_lang=source_lang
        )
        if translated_text:
            self.translation_queue.put(translated_text)

    def _toggle_recording(self):
        """録音ボタンのトグル動作（マイクストリーム＆文字起こしエンジンの開始・停止）"""
        if not self.is_recording:
            # 録音開始前の準備
            device_index = self._get_selected_device_index()
            model_name = parse_model_size(self.model_option.get())
            lang_code = parse_language_code(self.input_lang_option.get())

            self.status_label.configure(text=f"モデル準備中 ({model_name})...", text_color="#ffc107")
            self.record_button.configure(state="disabled")

            # モデルの初期化またはロードを別スレッドで実行してUIフリーズを回避
            threading.Thread(
                target=self._start_recording_thread,
                args=(device_index, model_name, lang_code),
                daemon=True,
            ).start()
        else:
            # 録音停止
            if self.processor:
                self.processor.stop()
            self.recorder.stop()
            self.is_recording = False

            # 停止時にバッファに残っているテキストを直ちにフラッシュ
            with self._buffer_lock:
                self._flush_translation_buffer_locked()

            # 話者登録をリセット（次回セッションを話者1から開始）
            self.diarizer.reset()

            self.record_button.configure(text="録音開始", fg_color="#28a745", hover_color="#218838")
            self.status_label.configure(text="ステータス: 停止中", text_color="gray")
            # 設定変更を再度有効化
            self.mic_option.configure(state="normal")
            self.model_option.configure(state="normal")
            self.input_lang_option.configure(state="normal")
            self.engine_option.configure(state="normal")

    def _save_current_settings(self, *args):
        """現在のUI設定を config.json に保存"""
        current_settings = {
            "transcription_model": self.model_option.get(),
            "translation_engine": self.engine_option.get(),
            "input_lang": self.input_lang_option.get(),
            "target_lang": self.target_lang_option.get(),
            "enable_separation": (self.sep_checkbox.get() == 1),
            "mic_device": self.mic_option.get(),
        }
        self.config_manager.save_config(current_settings)

    def _on_closing(self):
        """ウィンドウ終了時の処理"""
        try:
            self._save_current_settings()
            with self._buffer_lock:
                if self._buffer_timer is not None:
                    self._buffer_timer.cancel()
                    self._buffer_timer = None
                self._translation_buffer.clear()
            if self.processor:
                self.processor.stop()
            self.recorder.stop()
        except Exception:
            pass
        self.destroy()

    def _on_change_engine(self, selected_engine: str):
        """翻訳エンジンの切替ハンドラ"""
        if "ローカル" in selected_engine:
            nllb_model = "nllb-600M"
            if "1.3B" in selected_engine:
                nllb_model = "nllb-1.3B"
            elif "3.3B" in selected_engine:
                nllb_model = "nllb-3.3B"

            self.translator.set_nllb_model(nllb_model)
            self.translator.set_engine("local")

            if not (self.translator.local_translator and self.translator.local_translator._is_loaded):
                self.status_label.configure(
                    text=f"ローカル翻訳モデル ({nllb_model}) を準備中...",
                    text_color="#ffc107",
                )

                def load_local():
                    try:
                        self.translator.ensure_local_loaded()
                        self.after(
                            0,
                            lambda: self.status_label.configure(
                                text="ステータス: 準備完了 (ローカルAI翻訳)",
                                text_color="#28a745",
                            ),
                        )
                    except Exception as e:
                        err_msg = str(e)
                        self.after(
                            0,
                            lambda msg=err_msg: self.status_label.configure(
                                text=f"ローカル翻訳読込失敗: {msg}",
                                text_color="#dc3545",
                            ),
                        )

                threading.Thread(target=load_local, daemon=True).start()
            else:
                self.status_label.configure(
                    text="ステータス: 準備完了 (ローカルAI翻訳)",
                    text_color="#28a745",
                )
        else:
            self.translator.set_engine("cloud")
            self.status_label.configure(
                text="ステータス: 準備完了 (クラウド翻訳)",
                text_color="#28a745",
            )
        self._save_current_settings()

    def _on_toggle_separation(self):
        """話者識別（旧・音源分離）チェックボックスの切替ハンドラ"""
        is_enabled = (self.sep_checkbox.get() == 1)
        if self.processor:
            self.processor.enable_diarization = is_enabled

        if is_enabled and not self.diarizer.is_loaded:
            self.status_label.configure(text="3D-speaker話者識別モデルを読み込み中...", text_color="#ffc107")
            def load_thread():
                try:
                    self.diarizer.load_model()
                    self.after(0, lambda: self.status_label.configure(text="話者識別モデル準備完了", text_color="#28a745"))
                except Exception as e:
                    err_msg = str(e)
                    self.after(0, lambda msg=err_msg: self.status_label.configure(text=f"話者識別読込失敗: {msg}", text_color="#dc3545"))
            threading.Thread(target=load_thread, daemon=True).start()

        self._save_current_settings()

    def _start_recording_thread(self, device_index: int | None, model_name: str, lang_code: str | None):
        """録音および文字起こしのバックグラウンド初期化・開始"""
        try:
            # モデルのロードまたは切替
            if self.transcriber is None:
                self.transcriber = TranscriptionEngine(model_size=model_name)
            else:
                self.transcriber.change_model(model_size=model_name)

            # 音声プロセッサの初期化 (文単位のVAD処理 + 話者識別)
            self.processor = AudioProcessor(
                recorder=self.recorder,
                transcriber=self.transcriber,
                on_transcription_callback=self._on_transcription_received,
                diarizer=self.diarizer,
                separator=self.separator,
                silence_threshold=0.03,
                silence_duration_sec=0.8,
            )
            self.processor.enable_diarization = (self.sep_checkbox.get() == 1)

            # マイク録音開始
            self.recorder.start(device_index=device_index)
            # ワーカー開始
            self.processor.start(language=lang_code)

            self.is_recording = True

            # UIの更新をメインスレッドへディスパッチ
            def update_ui_on_success():
                self.record_button.configure(text="録音停止", fg_color="#dc3545", hover_color="#c82333", state="normal")
                self.status_label.configure(text="ステータス: 録音中...", text_color="#28a745")
                self.mic_option.configure(state="disabled")
                self.model_option.configure(state="disabled")
                self.input_lang_option.configure(state="disabled")
                self.engine_option.configure(state="disabled")

            self.after(0, update_ui_on_success)

        except Exception as e:
            err_msg = str(e)
            def update_ui_on_error(msg=err_msg):
                self.status_label.configure(text=f"エラー: 開始失敗 ({msg})", text_color="#dc3545")
                self.record_button.configure(state="normal")

            self.after(0, update_ui_on_error)

    def _copy_transcription(self):
        """文字起こしテキストのクリップボードコピー"""
        text = self.transcribe_textbox.get("1.0", tk.END).strip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            self._flash_button(self.copy_transcribe_btn, "コピー完了!")

    def _copy_translation(self):
        """翻訳テキストのクリップボードコピー"""
        text = self.translate_textbox.get("1.0", tk.END).strip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            self._flash_button(self.copy_translate_btn, "コピー完了!")

    def _flash_button(self, button: ctk.CTkButton, message: str):
        """ボタンのテキストを一時的に変更して戻すフィードバック"""
        original_text = button.cget("text")
        button.configure(text=message)
        self.after(1200, lambda: button.configure(text=original_text))

    def _clear_all_text(self):
        """全テキストエリアの内容を消去"""
        with self._buffer_lock:
            if self._buffer_timer is not None:
                self._buffer_timer.cancel()
                self._buffer_timer = None
            self._translation_buffer.clear()
        self.transcribe_textbox.delete("1.0", tk.END)
        self.translate_textbox.delete("1.0", tk.END)
        self.diarizer.reset()

    def append_transcription(self, text: str):
        """文字起こしテキストを末尾に追加し、自動スクロールする"""
        if not text:
            return
        self.transcribe_textbox.insert(tk.END, text + "\n")
        self.transcribe_textbox.see(tk.END)

    def append_translation(self, text: str):
        """翻訳テキストを末尾に追加し、自動スクロールする"""
        if not text:
            return
        self.translate_textbox.insert(tk.END, text + "\n")
        self.translate_textbox.see(tk.END)

