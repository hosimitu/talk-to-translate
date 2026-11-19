"""
ASR エンジンに依存しない共通ロジック。

CPU 推論スレッド数の決定と、文字起こし結果が語を含むかの判定を集約する。
"""

import os
import re

# 用途別の推論スレッド上限（ここで一元管理する）。
# 実測では 16 スレッド指定より 8 スレッドの方が高速だった
# （スレッド競合により 8 スレッド 179ms に対し 16 スレッド 208ms）。
TRANSCRIPTION_MAX_THREADS = 8  # SenseVoice / Whisper
DIARIZATION_MAX_THREADS = 4  # 話者識別

# 語を構成する文字（英数字・かな・漢字・ハングル等）。アンダースコアは語とみなさない。
_MEANINGFUL_TEXT_PATTERN = re.compile(r"[^\W_]", re.UNICODE)


def recommend_cpu_threads(max_threads: int = TRANSCRIPTION_MAX_THREADS) -> int:
    """
    CPU論理コア数から推論スレッド数の推奨値を求める。

    録音コールバックとUIスレッドをブロックしないよう全コアは使わず、
    論理コア数の半分（2コア以下では1）を上限 max_threads で頭打ちにする。
    max_threads が1以下なら常に1を返す。
    """
    cpu_count = os.cpu_count() or 2
    if cpu_count <= 2 or max_threads <= 1:
        return 1
    return min(max_threads, max(2, cpu_count // 2))


def is_meaningful_transcription(text: object) -> bool:
    """
    文字起こし結果が語を構成する文字を含むかを判定する。

    SenseVoice / Whisper は非発話音や環境音に対して "。" や "..." のような
    句読点・記号だけを出力することがある。これを弾いて無意味な表示や翻訳を防ぐ。
    """
    if not isinstance(text, str):
        return False
    return _MEANINGFUL_TEXT_PATTERN.search(text) is not None
