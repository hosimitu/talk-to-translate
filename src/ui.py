"""
Talk-to-Translate UIモジュール
CustomTkinterによるモダンなデスクトップGUIを提供します。
"""

import threading
import tkinter as tk
import customtkinter as ctk
import sounddevice as sd
from src.audio import AudioRecorder
from src.transcriber import TranscriptionEngine, parse_language_code, parse_model_size
from src.processor import AudioProcessor

# テーマ設定
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class AppUI(ctk.CTk):
    """メインアプリケーションウィンドウ"""

    def __init__(self):
        super().__init__()

        self.title("Talk-to-Translate")
        self.geometry("900x700")
        self.minsize(800, 600)

        # 音声レコーダーの初期化
        self.recorder = AudioRecorder(sample_rate=16000)

        # 文字起こしエンジン & プロセッサ
        self.transcriber: TranscriptionEngine | None = None
        self.processor: AudioProcessor | None = None

        # 録音中フラグ
        self.is_recording = False

        # グリッドのウェイト設定（中央のテキストエリアが伸縮）
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # UIコンポーネントの構築
        self._build_header_panel()
        self._build_text_areas()
        self._build_footer_panel()

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
        self.mic_option = ctk.CTkOptionMenu(header_frame, values=mics)
        self.mic_option.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        if mics:
            self.mic_option.set(mics[0])

        # 2. 文字起こしモデル選択
        model_label = ctk.CTkLabel(header_frame, text="AIモデル (Whisper):", font=ctk.CTkFont(size=12, weight="bold"))
        model_label.grid(row=0, column=1, padx=10, pady=(10, 0), sticky="w")
        
        models = ["tiny (最速・軽量)", "base (標準)", "small (高精度・要スペック)"]
        self.model_option = ctk.CTkOptionMenu(header_frame, values=models)
        self.model_option.grid(row=1, column=1, padx=10, pady=(0, 10), sticky="ew")
        self.model_option.set("base (標準)")

        # 3. 入力音声言語
        lang_label = ctk.CTkLabel(header_frame, text="入力言語 (音声):", font=ctk.CTkFont(size=12, weight="bold"))
        lang_label.grid(row=0, column=2, padx=10, pady=(10, 0), sticky="w")
        
        languages = ["自動検出 (auto)", "日本語 (ja)", "英語 (en)", "中国語 (zh)"]
        self.input_lang_option = ctk.CTkOptionMenu(header_frame, values=languages)
        self.input_lang_option.grid(row=1, column=2, padx=10, pady=(0, 10), sticky="ew")
        self.input_lang_option.set("自動検出 (auto)")

        # 4. 翻訳先言語 & 翻訳有効スイッチ
        trans_label = ctk.CTkLabel(header_frame, text="翻訳先言語:", font=ctk.CTkFont(size=12, weight="bold"))
        trans_label.grid(row=0, column=3, padx=10, pady=(10, 0), sticky="w")
        
        target_languages = ["英語 (en)", "日本語 (ja)", "中国語 (zh)"]
        self.target_lang_option = ctk.CTkOptionMenu(header_frame, values=target_languages)
        self.target_lang_option.grid(row=1, column=3, padx=10, pady=(0, 10), sticky="ew")
        self.target_lang_option.set("英語 (en)")

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

        # 文字起こしヘッダー (ラベル & コピーボタン)
        transcribe_header = ctk.CTkFrame(left_frame, fg_color="transparent")
        transcribe_header.grid(row=0, column=0, padx=15, pady=(10, 5), sticky="ew")
        transcribe_header.grid_columnconfigure(0, weight=1)

        transcribe_title = ctk.CTkLabel(
            transcribe_header, 
            text="🎙️ 文字起こし結果", 
            font=ctk.CTkFont(size=14, weight="bold")
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

        # 文字起こしテキストボックス
        self.transcribe_textbox = ctk.CTkTextbox(left_frame, wrap="word", font=ctk.CTkFont(size=13))
        self.transcribe_textbox.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="nsew")

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
            font=ctk.CTkFont(size=14, weight="bold")
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

        # 翻訳テキストボックス
        self.translate_textbox = ctk.CTkTextbox(right_frame, wrap="word", font=ctk.CTkFont(size=13))
        self.translate_textbox.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="nsew")

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
        """バックグラウンドスレッドからの文字起こし結果を受け取りUIへ反映"""
        # メインスレッドでUI更新
        self.after(0, self.append_transcription, text)

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

            self.record_button.configure(text="録音開始", fg_color="#28a745", hover_color="#218838")
            self.status_label.configure(text="ステータス: 停止中", text_color="gray")
            # 設定変更を再度有効化
            self.mic_option.configure(state="normal")
            self.model_option.configure(state="normal")
            self.input_lang_option.configure(state="normal")

    def _start_recording_thread(self, device_index: int | None, model_name: str, lang_code: str | None):
        """録音および文字起こしのバックグラウンド初期化・開始"""
        try:
            # モデルのロードまたは切替
            if self.transcriber is None:
                self.transcriber = TranscriptionEngine(model_size=model_name)
            else:
                self.transcriber.change_model(model_size=model_name)

            # 音声プロセッサの初期化
            self.processor = AudioProcessor(
                recorder=self.recorder,
                transcriber=self.transcriber,
                on_transcription_callback=self._on_transcription_received,
                chunk_duration_sec=3.0,
            )

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

            self.after(0, update_ui_on_success)

        except Exception as e:
            def update_ui_on_error():
                self.status_label.configure(text=f"エラー: 開始失敗 ({e})", text_color="#dc3545")
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
        self.transcribe_textbox.delete("1.0", tk.END)
        self.translate_textbox.delete("1.0", tk.END)

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

