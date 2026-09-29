"""SenseVoice ONNX の推論テスト"""

import huggingface_hub
import numpy as np
import sherpa_onnx


def test_sensevoice_inference():
    print("SenseVoice ONNXモデルをロード中...")
    model_path = huggingface_hub.hf_hub_download(
        repo_id="csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
        filename="model.int8.onnx",
    )
    tokens_path = huggingface_hub.hf_hub_download(
        repo_id="csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
        filename="tokens.txt",
    )

    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=model_path,
        tokens=tokens_path,
        num_threads=2,
        use_itn=True,
    )

    # 1秒分のダミー音声
    dummy_audio = np.zeros(16000, dtype=np.float32)
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, dummy_audio)
    recognizer.decode_stream(stream)
    text = stream.result.text.strip()

    print(f"[OK] SenseVoice推論成功 (出力: '{text}')")


if __name__ == "__main__":
    test_sensevoice_inference()
