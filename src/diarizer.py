"""
Talk-to-Translate 話者ダイアライゼーション（話者識別）モジュール
sherpa-onnx (3D-speaker) を使用して、複数人の発話を「話者1」「話者2」「話者3」...と超軽量・高精度に自動識別します。
"""

import threading
from typing import Dict, List, Optional
import huggingface_hub
import numpy as np
import sherpa_onnx


def should_update_profile(match_count: int, interval: int) -> bool:
    """
    一致した話者のプロファイル（埋め込み）を更新すべきか判定する。

    毎回の一致で更新すると相槌や雑音でプロファイルが歪むため、一定回数ごとに間引く。
    :param match_count: その話者と一致した回数（1始まり）
    :param interval: 更新間隔（1以下なら毎回更新）
    """
    if match_count <= 0:
        return False
    if interval <= 1:
        return True
    return match_count % interval == 0


class SpeakerDiarizer:
    """3D-speaker (ONNX) を用いたリアルタイム話者識別クラス"""

    REPO_ID = "csukuangfj/speaker-embedding-models"
    DEFAULT_MODEL = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

    def __init__(
        self,
        threshold: float = 0.55,
        num_threads: int = 2,
        model_filename: Optional[str] = None,
        min_embedding_duration_sec: float = 0.8,
        profile_update_interval: int = 3,
        max_speakers: int = 8,
    ):
        """
        初期化
        :param threshold: コサイン類似度の閾値 (0.50〜0.60推奨、デフォルト0.55)
        :param num_threads: CPU推論スレッド数
        :param model_filename: 使用する3D-speakerモデルファイル名 (NoneでデフォルトERes2Net)
        :param min_embedding_duration_sec: プロファイル学習・新規登録を行う最小発話長（秒）
        :param profile_update_interval: 同一話者のプロファイルを更新する間隔（一致回数）
        :param max_speakers: 登録する話者の上限（超過時は新規登録しない）
        """
        self.threshold = threshold
        self.num_threads = num_threads
        self.model_filename = model_filename or self.DEFAULT_MODEL
        self.min_embedding_duration_sec = min_embedding_duration_sec
        self.profile_update_interval = profile_update_interval
        self.max_speakers = max_speakers

        self._extractor: Optional[sherpa_onnx.SpeakerEmbeddingExtractor] = None
        self._manager: Optional[sherpa_onnx.SpeakerEmbeddingManager] = None
        self._is_loaded = False
        self._lock = threading.Lock()
        self._match_counts: Dict[str, int] = {}

    @property
    def is_loaded(self) -> bool:
        """モデルがロード済みかどうか"""
        return self._is_loaded

    @property
    def num_speakers(self) -> int:
        """現在登録されている話者数"""
        with self._lock:
            if self._manager is not None:
                return self._manager.num_speakers
            return 0

    @property
    def all_speakers(self) -> List[str]:
        """現在登録されている全話者名リスト"""
        with self._lock:
            if self._manager is not None:
                return list(self._manager.all_speakers)
            return []

    def load_model(self):
        """モデルをオンデマンドでロード"""
        with self._lock:
            if self._is_loaded:
                return

            print(f"[SpeakerDiarizer] 3D-speaker モデル ({self.model_filename}) をロード中...")
            model_path = huggingface_hub.hf_hub_download(
                repo_id=self.REPO_ID,
                filename=self.model_filename,
            )

            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=model_path,
                num_threads=self.num_threads,
                provider="cpu",
            )
            if not config.validate():
                raise RuntimeError("SpeakerEmbeddingExtractorConfig の検証に失敗しました。")

            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
            self._manager = sherpa_onnx.SpeakerEmbeddingManager(self._extractor.dim)
            self._is_loaded = True
            print(f"[SpeakerDiarizer] モデルのロードが完了しました。(埋め込み次元: {self._extractor.dim})")

    def identify_speaker(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Optional[str]:
        """
        音声データから話者を識別し、話者ラベル ('話者1', '話者2'...) を返す。
        新規話者の場合は自動登録して新しいラベルを採番する。
        既存話者と判定された場合は埋め込みを追加学習して認識精度を向上させる。
        :param audio_data: 16000Hz float32 音声信号
        :param sample_rate: サンプリングレート
        :return: 話者ラベル ('話者1' など) または 識別不能時は None
        """
        if audio_data is None or len(audio_data) == 0:
            return None

        if not self._is_loaded:
            self.load_model()

        # 短すぎる発話（相槌・雑音）ではプロファイルを学習・新規登録しない
        duration_sec = len(audio_data) / float(sample_rate) if sample_rate > 0 else 0.0
        can_learn = duration_sec >= self.min_embedding_duration_sec

        with self._lock:
            if self._extractor is None or self._manager is None:
                return None

            try:
                stream = self._extractor.create_stream()
                stream.accept_waveform(sample_rate, audio_data)
                stream.input_finished()

                if not self._extractor.is_ready(stream):
                    return None

                embedding = self._extractor.compute(stream)
                matched_speaker = self._manager.search(embedding, threshold=self.threshold)

                if matched_speaker:
                    # 一致回数を数え、一定間隔でのみプロファイルを更新する
                    count = self._match_counts.get(matched_speaker, 0) + 1
                    self._match_counts[matched_speaker] = count
                    if can_learn and should_update_profile(count, self.profile_update_interval):
                        self._manager.add(matched_speaker, embedding)
                    return matched_speaker

                # 短い発話は新規話者として登録しない（誤登録・番号増加の抑制）
                if not can_learn:
                    return None

                # 上限に達した場合は新規登録せず、話者タグなしで返す
                if self._manager.num_speakers >= self.max_speakers:
                    return None

                # 未知の話者のため新規登録
                next_index = self._manager.num_speakers + 1
                new_speaker_name = f"話者{next_index}"
                self._manager.add(new_speaker_name, embedding)
                self._match_counts[new_speaker_name] = 1
                return new_speaker_name

            except Exception as e:
                print(f"[SpeakerDiarizer Error] 話者識別処理に失敗: {e}")
                return None

    def reset(self):
        """登録話者リストをリセット（次の会話を話者1から開始する）"""
        with self._lock:
            if self._extractor is not None:
                self._manager = sherpa_onnx.SpeakerEmbeddingManager(self._extractor.dim)
            self._match_counts.clear()
            print("[SpeakerDiarizer] 登録話者情報をリセットしました。")
