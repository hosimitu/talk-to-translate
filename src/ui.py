import queue
import re
import threading
import time
import tkinter as tk
from typing import Optional
import customtkinter as ctk
import sounddevice as sd
from src.config import (
    ConfigManager,
    DEFAULT_CONFIG,
    INPUT_LANGUAGES,
    PRIVACY_NOTICE_VERSION,
    TARGET_LANGUAGES,
    TRANSCRIPTION_MODELS,
    TRANSLATION_ENGINES,
)
from src.audio import AudioRecorder
from src.transcriber import (
    TranscriptionEngine,
    detect_acceleration,
    parse_language_code,
    parse_model_size,
)
from src.processor import AudioProcessor
from src.diarizer import SpeakerDiarizer
from src.translator import (
    TextTranslator,
    parse_target_language_code,
    parse_source_language_code,
    parse_translation_engine,
)

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

        # 話者識別エンジン
        self.diarizer = SpeakerDiarizer()

        # 翻訳エンジンの初期化（保存された設定を反映）
        saved_engine = self.config.get("translation_engine", "")
        engine, nllb_model = parse_translation_engine(saved_engine)

        self.translator = TextTranslator(
            source_lang="auto",
            engine=engine,
            nllb_model=nllb_model,
        )

        # スレッド間通信用キュー
        self.transcription_queue: queue.Queue[tuple[str, Optional[str]]] = queue.Queue()
        self.translation_queue: queue.Queue[tuple[int, str]] = queue.Queue()
        # 翻訳は単一ワーカーで順番に処理する（応答順の入れ替わりを防ぐ）
        self._translation_requests: queue.Queue[Optional[tuple[int, list[str], str, str]]] = queue.Queue()
        self._translation_worker: Optional[threading.Thread] = None
        self._translation_generation = 0  # 全消去時に加算し、古い翻訳結果を破棄する

        # 録音中フラグ
        self.is_recording = False
        self._last_activity_update = 0.0

        # 翻訳用小バッファリング管理 (2〜3行 / 約70文字 / 2.0秒休止フラッシュ)
        self._translation_buffer: list[str] = []
        self._buffer_lock = threading.Lock()
        self._buffer_timer: Optional[threading.Timer] = None
        self._buffer_flush_delay = 2.0
        self._buffer_max_lines = 3
        self._buffer_max_chars = 70

        # 翻訳言語コードのキャッシュ（ワーカースレッドからTkウィジェットを参照しないため）
        self._source_lang = "auto"
        self._target_lang = "en"
        # 直近の発話から検出された言語（入力言語が「自動検出」のときに使用）
        self._last_source_lang: Optional[str] = None

        # グリッドのウェイト設定（中央のテキストエリアが伸縮）
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # UIコンポーネントの構築
        self._build_header_panel()
        self._build_text_areas()
        self._build_footer_panel()
        self._sync_language_codes()

        # ウィンドウ終了時のハンドラ登録
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        # 翻訳専用ワーカーを起動（結果の順序保証）
        self._start_translation_worker()

        # キューの監視ループを開始
        self._poll_queues()

    def _start_translation_worker(self):
        """翻訳専用の単一ワーカースレッドを起動する"""
        if self._translation_worker is not None and self._translation_worker.is_alive():
            return
        self._translation_worker = threading.Thread(
            target=self._translation_worker_loop,
            name="translation-worker",
            daemon=True,
        )
        self._translation_worker.start()

    def _translation_worker_loop(self):
        """キューに積まれた翻訳ジョブを到着順に処理する"""
        while True:
            job = self._translation_requests.get()
            if job is None:  # 終了シグナル
                break

            generation, lines, target_lang, source_lang = job
            try:
                translated = self._translate_buffered_lines(lines, target_lang, source_lang)
            except Exception as e:
                print(f"[UI] 翻訳処理に失敗しました: {e}")
                continue

            if not translated:
                continue
            # 全消去などで世代が進んでいたら古い結果は捨てる
            if generation != self._translation_generation:
                continue
            self.translation_queue.put((generation, translated))

    def _poll_queues(self):
        """ワーカースレッドからのキューをメインスレッドで安全に処理"""
        self._drain_transcription_queue()
        self._drain_translation_queue()

        # 50ミリ秒ごとに再帰チェック
        self.after(50, self._poll_queues)

    def _drain_transcription_queue(self):
        """文字起こし結果をテキストボックスへ反映し、検出言語を記録する"""
        while True:
            try:
                item = self.transcription_queue.get_nowait()
            except queue.Empty:
                break

            if isinstance(item, tuple) and len(item) == 2:
                text, lang = item
                if lang:
                    self._last_source_lang = lang
            else:
                text = item

            self.append_transcription(text)

    def _drain_translation_queue(self):
        """翻訳結果のうち、現在の世代のものだけを反映する"""
        while True:
            try:
                item = self.translation_queue.get_nowait()
            except queue.Empty:
                break

            if isinstance(item, tuple) and len(item) == 2:
                generation, text = item
                if generation != self._translation_generation:
                    continue
            else:
                text = item

            self.append_translation(text)

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

        # 保存されていたマイクデバイスの復元 (デバイス番号のズレを吸収して名前で照合)
        saved_mic = self.config.get("mic_device", "")
        saved_name = saved_mic.split(":", 1)[1].strip() if ":" in saved_mic else saved_mic.strip()
        matched_mic = None
        if saved_name:
            matched_mic = next((m for m in mics if saved_name.lower() in m.lower()), None)

        if matched_mic:
            self.mic_option.set(matched_mic)
        elif mics:
            self.mic_option.set(mics[0])

        # 2. 文字起こしモデル選択
        model_label = ctk.CTkLabel(header_frame, text="AIモデル:", font=ctk.CTkFont(size=12, weight="bold"))
        model_label.grid(row=0, column=1, padx=10, pady=(10, 0), sticky="w")
        
        models = TRANSCRIPTION_MODELS
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
            self.model_option.set(DEFAULT_CONFIG["transcription_model"])

        # 3. 入力音声言語
        lang_label = ctk.CTkLabel(header_frame, text="入力言語 (音声):", font=ctk.CTkFont(size=12, weight="bold"))
        lang_label.grid(row=0, column=2, padx=10, pady=(10, 0), sticky="w")
        
        languages = INPUT_LANGUAGES
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
            self.input_lang_option.set(DEFAULT_CONFIG["input_lang"])

        # 4. 翻訳先言語
        trans_label = ctk.CTkLabel(header_frame, text="翻訳先言語:", font=ctk.CTkFont(size=12, weight="bold"))
        trans_label.grid(row=0, column=3, padx=10, pady=(10, 0), sticky="w")
        
        target_languages = TARGET_LANGUAGES
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
            self.target_lang_option.set(DEFAULT_CONFIG["target_lang"])

        # 5. 翻訳エンジン選択 (2行目の前半)
        engine_label = ctk.CTkLabel(header_frame, text="翻訳エンジン:", font=ctk.CTkFont(size=12, weight="bold"))
        engine_label.grid(row=2, column=0, padx=10, pady=(6, 12), sticky="w")

        engines = TRANSLATION_ENGINES
        self.engine_option = ctk.CTkOptionMenu(
            header_frame,
            values=engines,
            command=self._on_change_engine,
        )
        self.engine_option.grid(row=2, column=1, padx=10, pady=(6, 12), sticky="ew")

        # 保存されていた翻訳エンジンの復元（旧表記は ConfigManager 側で正規化済み）
        saved_engine = self.config.get("translation_engine")
        if saved_engine in engines:
            self.engine_option.set(saved_engine)
        else:
            self.engine_option.set(DEFAULT_CONFIG["translation_engine"])

        # 6. 話者識別チェックボックス (2行目の後半に配置)
        self.diarization_checkbox = ctk.CTkCheckBox(
            header_frame,
            text="👥 話者識別（3D-speaker ONNX / 多人数対応）",
            font=ctk.CTkFont(size=12),
            command=self._on_toggle_diarization,
        )
        self.diarization_checkbox.grid(row=2, column=2, columnspan=2, padx=15, pady=(6, 12), sticky="w")

        # 保存されていた話者識別設定の復元
        if self.config.get("enable_diarization", False):
            self.diarization_checkbox.select()
        else:
            self.diarization_checkbox.deselect()

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

        # プライバシー表示（クラウド翻訳時の外部送信を常時明示）
        self.privacy_badge = ctk.CTkLabel(
            footer_frame,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        self.privacy_badge.grid(row=1, column=0, columnspan=2, padx=20, pady=(0, 10), sticky="w")

        # 文字起こしモデル／推論デバイス表示（GPU自動判定の結果を可視化）
        self.model_badge = ctk.CTkLabel(
            footer_frame,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        self.model_badge.grid(row=1, column=2, padx=20, pady=(0, 10), sticky="e")

        # 初回起動時の通知バナー（モーダルにせず、その場で閉じられる）
        self.privacy_notice = ctk.CTkFrame(
            footer_frame,
            fg_color=("#e7f1ff", "#1f3a5f"),
            corner_radius=8,
        )
        self.privacy_notice.grid(row=2, column=0, columnspan=3, padx=15, pady=(0, 12), sticky="ew")
        self.privacy_notice.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.privacy_notice,
            text=(
                "☁ クラウド翻訳を選択すると、翻訳対象のテキストのみが外部（Google / MyMemory）へ"
                "送信されます。音声そのものは送信されません。完全オフラインで使う場合は"
                "「ローカル翻訳 (NLLB)」を選択してください。"
            ),
            font=ctk.CTkFont(size=11),
            justify="left",
            wraplength=680,
        ).grid(row=0, column=0, padx=12, pady=10, sticky="w")

        ctk.CTkButton(
            self.privacy_notice,
            text="了解",
            width=60,
            height=26,
            command=self._acknowledge_privacy_notice,
        ).grid(row=0, column=1, padx=12, pady=10, sticky="e")

        self._update_privacy_badge()
        self._update_model_badge()
        self._refresh_privacy_notice()

    def _current_translation_engine_kind(self) -> str:
        """現在選択されている翻訳エンジン種別 ("cloud" / "local") を返す"""
        engine, _ = parse_translation_engine(self.engine_option.get())
        return engine

    def _update_privacy_badge(self):
        """翻訳エンジンに応じて外部送信の有無を表示する"""
        if self._current_translation_engine_kind() == "cloud":
            self.privacy_badge.configure(
                text="☁ 外部送信: 有効（翻訳テキストのみ送信）",
                text_color="#ffc107",
            )
        else:
            self.privacy_badge.configure(
                text="🔒 ローカルのみ（外部送信なし）",
                text_color="#28a745",
            )

    def _update_model_badge(self):
        """文字起こしモデルと推論デバイス（GPU/CPU）を表示する"""
        model_name = parse_model_size(self.model_option.get())
        if model_name == "sensevoice":
            self.model_badge.configure(text="⚙ SenseVoice (CPU)", text_color="gray")
            return

        device = getattr(self.transcriber, "device", None)
        if not device:
            device, _ = detect_acceleration()
        if device == "cuda":
            self.model_badge.configure(text="⚙ Whisper (GPU)", text_color="#28a745")
        else:
            self.model_badge.configure(text="⚙ Whisper (CPU)", text_color="gray")

    def _refresh_privacy_notice(self):
        """同意済みなら通知バナーを隠す"""
        if self._get_privacy_ack_version() >= PRIVACY_NOTICE_VERSION:
            self.privacy_notice.grid_remove()
        else:
            self.privacy_notice.grid()

    def _acknowledge_privacy_notice(self):
        """プライバシー通知を確認済みとして保存し、バナーを閉じる"""
        self.config["privacy_ack_version"] = PRIVACY_NOTICE_VERSION
        self._save_current_settings()
        self.privacy_notice.grid_remove()

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
        # 左側の文字起こしテキストボックスへは即座に反映（検出言語も併せて渡す）
        detected_lang = getattr(self.transcriber, "last_detected_language", None)
        self.transcription_queue.put((text, detected_lang))

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

        # Tkウィジェットはメインスレッド専用のため、キャッシュ済みの言語コードを使う。
        # 入力言語が「自動検出」の場合は、直近の発話で検出された言語を優先する。
        source_lang = self._source_lang
        if source_lang == "auto" and self._last_source_lang:
            source_lang = self._last_source_lang
        target_lang = self._target_lang
        self._translation_requests.put(
            (self._translation_generation, buffered_lines, target_lang, source_lang)
        )

    def _translate_buffered_lines(self, lines: list[str], target_lang: str, source_lang: str = "auto") -> str:
        """バッファされた複数行を自然に結合して翻訳する（単一ワーカーから呼ばれる）"""
        if not lines:
            return ""

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
            return ""

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

        return self.translator.translate(
            combined_text, target_lang=target_lang, source_lang=source_lang
        )

    def _toggle_recording(self):
        """録音ボタンのトグル動作（マイクストリーム＆文字起こしエンジンの開始・停止）"""
        if not self.is_recording:
            # 録音開始前の準備
            device_index = self._get_selected_device_index()
            model_name = parse_model_size(self.model_option.get())
            lang_code = parse_language_code(self.input_lang_option.get())
            self._last_source_lang = None

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

    def _sync_language_codes(self):
        """UIの言語選択を翻訳用コードへ変換してキャッシュ（メインスレッドで実行）"""
        self._source_lang = parse_source_language_code(self.input_lang_option.get())
        self._target_lang = parse_target_language_code(self.target_lang_option.get())

    def _get_privacy_ack_version(self) -> int:
        """保存済みのプライバシー同意バージョンを取得（不正値は0）"""
        value = self.config.get("privacy_ack_version", 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return 0
        return value

    def _save_current_settings(self, *args):
        """現在のUI設定を config.json に保存"""
        self._sync_language_codes()
        current_settings = {
            "transcription_model": self.model_option.get(),
            "translation_engine": self.engine_option.get(),
            "input_lang": self.input_lang_option.get(),
            "target_lang": self.target_lang_option.get(),
            "enable_diarization": (self.diarization_checkbox.get() == 1),
            "mic_device": self.mic_option.get(),
            "privacy_ack_version": self._get_privacy_ack_version(),
        }
        self.config_manager.save_config(current_settings)
        self._update_model_badge()

    def _on_closing(self):
        """ウィンドウ終了時の処理"""
        try:
            self._save_current_settings()
            with self._buffer_lock:
                if self._buffer_timer is not None:
                    self._buffer_timer.cancel()
                    self._buffer_timer = None
                self._translation_buffer.clear()
            # 翻訳ワーカーを停止
            self._translation_requests.put(None)
            if self._translation_worker is not None:
                self._translation_worker.join(timeout=1.0)
            if self.processor:
                self.processor.stop()
            self.recorder.stop()
        except Exception:
            pass
        self.destroy()

    def _on_change_engine(self, selected_engine: str):
        """翻訳エンジンの切替ハンドラ"""
        engine, nllb_model = parse_translation_engine(selected_engine)
        if engine == "local":
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
        self._update_privacy_badge()
        self._save_current_settings()

    def _on_toggle_diarization(self):
        """話者識別チェックボックスの切替ハンドラ"""
        is_enabled = (self.diarization_checkbox.get() == 1)
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
                silence_threshold=0.006,
                silence_duration_sec=0.8,
                min_speech_duration_sec=0.3,
                on_activity_callback=self._on_audio_activity,
            )
            self.processor.enable_diarization = (self.diarization_checkbox.get() == 1)

            # マイク録音開始
            self.recorder.start(device_index=device_index)
            # ワーカー開始
            self.processor.start(language=lang_code)

            self.is_recording = True

            # UIの更新をメインスレッドへディスパッチ
            def update_ui_on_success():
                self.record_button.configure(text="録音停止", fg_color="#dc3545", hover_color="#c82333", state="normal")
                self.status_label.configure(text="ステータス: 録音中 (音声待機中...)", text_color="#17a2b8")
                self.mic_option.configure(state="disabled")
                self.model_option.configure(state="disabled")
                self.input_lang_option.configure(state="disabled")
                self.engine_option.configure(state="disabled")
                self._update_model_badge()

            self.after(0, update_ui_on_success)

        except Exception as e:
            err_msg = str(e)
            def update_ui_on_error(msg=err_msg):
                self.status_label.configure(text=f"エラー: 開始失敗 ({msg})", text_color="#dc3545")
                self.record_button.configure(state="normal")

            self.after(0, update_ui_on_error)

    def _on_audio_activity(self, rms: float, is_active: bool):
        """マイク音声の入力レベルおよび発話検知状態のUI反映"""
        if not self.is_recording:
            return

        now = time.time()
        # UI更新頻度を制限 (250msごと)
        if now - self._last_activity_update < 0.25:
            return
        self._last_activity_update = now

        def update():
            if not self.is_recording:
                return
            if is_active:
                self.status_label.configure(text="ステータス: 録音中 (🎤 音声検知中...)", text_color="#28a745")
            elif rms < 0.0002:
                self.status_label.configure(text="ステータス: 録音中 (⚠️ 入力なし: マイク設定を確認してください)", text_color="#ffc107")
            else:
                self.status_label.configure(text="ステータス: 録音中 (音声待機中...)", text_color="#17a2b8")

        self.after(0, update)

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
        """全テキストエリアの内容を消去（実行中の翻訳結果も破棄）"""
        with self._buffer_lock:
            if self._buffer_timer is not None:
                self._buffer_timer.cancel()
                self._buffer_timer = None
            self._translation_buffer.clear()
            # 世代を進め、処理中の翻訳が後から復活しないようにする
            self._translation_generation += 1

        # 未反映の翻訳結果を破棄
        while True:
            try:
                self.translation_queue.get_nowait()
            except queue.Empty:
                break

        self._last_source_lang = None
        self.transcribe_textbox.delete("1.0", tk.END)
        self.translate_textbox.delete("1.0", tk.END)
        self.diarizer.reset()

    @staticmethod
    def _is_scrolled_to_bottom(textbox: ctk.CTkTextbox) -> bool:
        """テキストボックスが最下部に表示されているか"""
        try:
            return textbox.yview()[1] >= 0.999
        except Exception:
            return True

    def append_transcription(self, text: str):
        """文字起こしテキストを末尾に追加し、最下部にいるときだけ自動スクロールする"""
        if not text:
            return
        at_bottom = self._is_scrolled_to_bottom(self.transcribe_textbox)
        self.transcribe_textbox.insert(tk.END, text + "\n")
        if at_bottom:
            self.transcribe_textbox.see(tk.END)

    def append_translation(self, text: str):
        """翻訳結果を末尾に追加し、最下部にいるときだけ自動スクロールする"""
        if not text:
            return
        at_bottom = self._is_scrolled_to_bottom(self.translate_textbox)
        self.translate_textbox.insert(tk.END, text + "\n")
        if at_bottom:
            self.translate_textbox.see(tk.END)

