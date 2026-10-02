"""
Talk-to-Translate 音源分離モジュール
Alibaba ClearVoice (MossFormer2_SS_16K) を使用して、複数人が同時に話した音声を独立したトラックに分離します。
"""

import os
import tempfile
from typing import Optional
import numpy as np


class SpeechSeparator:
    """MossFormer2を用いた音声分離エンジン"""

    def __init__(self):
        self._cv = None
        self._is_loaded = False

    def load_model(self):
        """モデルをオンデマンドでロード（初回ON時のみ実行）"""
        if self._is_loaded:
            return

        from clearvoice import ClearVoice
        print("[SpeechSeparator] MossFormer2_SS_16K モデルを読み込んでいます...")
        self._cv = ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"])
        self._is_loaded = True
        print("[SpeechSeparator] モデルの読み込みが完了しました。")

    def separate(self, audio_data: np.ndarray, sample_rate: int = 16000) -> list[np.ndarray]:
        """
        音声を2人の話者に分離する
        :param audio_data: 16000Hz float32 音声信号
        :param sample_rate: サンプリングレート
        :return: 分離された音声トラックのリスト [track_1, track_2]
        """
        if not self._is_loaded:
            try:
                self.load_model()
            except Exception as e:
                print(f"[SpeechSeparator Error] モデルロード失敗のため元音声を返します: {e}")
                return [audio_data]

        if self._cv is None or len(audio_data) == 0:
            return [audio_data]

        try:
            import soundfile as sf
        except ImportError as e:
            print(f"[SpeechSeparator Error] soundfile がインポートできないため音源分離をスキップします: {e}")
            return [audio_data]

        temp_in = None
        temp_out_dir = None
        try:
            # 入力用の一時WAVファイルを作成
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
                temp_in = f_in.name
                sf.write(temp_in, audio_data, sample_rate)

            temp_out_dir = tempfile.mkdtemp()

            # ClearVoiceによる音源分離を実行
            self._cv(input_path=temp_in, online_write=True, output_path=temp_out_dir)

            # 出力されたwavファイルを取得
            separated_tracks = []
            for file_name in sorted(os.listdir(temp_out_dir)):
                if file_name.endswith(".wav"):
                    file_path = os.path.join(temp_out_dir, file_name)
                    track_data, _ = sf.read(file_path, dtype="float32")
                    # エネルギーが極端に小さい無音トラックは除外
                    if np.max(np.abs(track_data)) > 0.02:
                        separated_tracks.append(track_data)

            return separated_tracks if separated_tracks else [audio_data]

        except Exception as e:
            print(f"[SpeechSeparator Error] 音源分離処理失敗: {e}")
            return [audio_data]

        finally:
            # 一時ファイルの削除
            if temp_in and os.path.exists(temp_in):
                try:
                    os.remove(temp_in)
                except Exception:
                    pass
            if temp_out_dir and os.path.exists(temp_out_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_out_dir, ignore_errors=True)
                except Exception:
                    pass
