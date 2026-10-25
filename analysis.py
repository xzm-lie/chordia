"""Bounded audio decoding and reusable signal analysis; no UI or network calls."""
from __future__ import annotations

import io
import math
import wave
from typing import Callable

import librosa
import numpy as np
import soundfile as sf
import soxr

SR = 22050
HOP = 512
N_FFT = 2048
MAX_BYTES = 100 * 1024 * 1024
NOTES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


class AudioError(ValueError):
    """An actionable, safe-to-display input error."""


def decode_audio(data: bytes, offset: float, seconds: int):
    if not data:
        raise AudioError("文件为空，请重新选择音频。")
    if len(data) > MAX_BYTES:
        raise AudioError("文件超过 50 MB，请先裁剪或压缩。")
    if not math.isfinite(offset) or offset < 0 or not 1 <= seconds <= 300:
        raise AudioError("分析起点必须非负，分析时长须为 1–300 秒。")
    try:
        with sf.SoundFile(io.BytesIO(data)) as source:
            total = source.frames / source.samplerate
            if source.samplerate > 192000 or source.channels > 8:
                raise AudioError("请使用采样率不高于 192 kHz、声道数不超过 8 的音频。")
            start = round(offset * source.samplerate)
            if start >= source.frames:
                raise AudioError("分析起点超出了音频时长，请调小起点。")
            source.seek(start)
            remaining = min(source.frames - start, round(seconds * source.samplerate))
            if remaining / source.samplerate < 0.25:
                raise AudioError("可分析片段不足 0.25 秒，请选择更长的片段。")
            resampler = soxr.ResampleStream(source.samplerate, SR, 1, dtype="float32")
            chunks = []
            while remaining:
                block = source.read(min(65536, remaining), dtype="float32", always_2d=True)
                if not len(block):
                    raise AudioError("音频数据提前结束，文件可能已损坏。")
                remaining -= len(block)
                mono = block.mean(axis=1)
                chunks.append(resampler.resample_chunk(mono, last=remaining == 0))
            y = np.concatenate(chunks).astype(np.float32, copy=False)
            metadata = {"total_duration": total, "source_sr": source.samplerate,
                        "channels": source.channels, "offset": start / source.samplerate}
    except AudioError:
        raise
    except (RuntimeError, ValueError, OSError) as exc:
        raise AudioError("无法解码音频。请确认文件完整，或将它转换成标准 WAV / MP3 / FLAC。") from exc
    if not np.isfinite(y).all():
        raise AudioError("音频包含无效采样值，请重新导出文件。")
    return y, metadata


def key_estimate(chroma_mean: np.ndarray) -> dict:
    x = np.asarray(chroma_mean, dtype=float)
    x = x - x.mean()
    norm = np.linalg.norm(x)
    if norm < 1e-8:
        return {"label": "暂无法判断", "score": 0.0, "margin": 0.0, "alternatives": []}
    candidates = []
    for root, note in enumerate(NOTES):
        for mode, profile in (("大调", MAJOR), ("小调", MINOR)):
            profile = np.roll(profile, root)
            profile = profile - profile.mean()
            score = float(np.dot(x, profile) / (norm * np.linalg.norm(profile)))
            candidates.append({"label": f"{note} {mode}", "score": score})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    top = candidates[0]
    return {"label": top["label"] if top["score"] >= 0.3 else "暂无法判断",
            "score": top["score"], "margin": top["score"] - candidates[1]["score"],
            "alternatives": candidates[:3]}


def chord_segments(chroma, rms, duration, window, threshold) -> list[dict]:
    templates, names = [], []
    for root, note in enumerate(NOTES):
        for third, suffix in ((4, ""), (3, "m")):
            template = np.zeros(12, dtype=np.float32)
            template[[root, (root + third) % 12, (root + 7) % 12]] = 1 / np.sqrt(3)
            templates.append(template)
            names.append(note + suffix)
    templates = np.stack(templates)
    step = max(1, round(window * SR / HOP))
    merged = []
    for start in range(0, chroma.shape[1], step):
        t0 = start * HOP / SR
        if t0 >= duration:
            break
        stop = min(start + step, chroma.shape[1])
        t1 = min(stop * HOP / SR, duration)
        active = rms[start:stop] > threshold
        label, score = "N", 0.0
        if active.mean() >= 0.3:
            vector = chroma[:, start:stop][:, active].mean(axis=1)
            norm = np.linalg.norm(vector)
            if norm > 1e-8:
                scores = templates @ (vector / norm)
                best = int(np.argmax(scores))
                score = float(scores[best])
                runner_up = float(np.partition(scores, -2)[-2])
                if score >= 0.62 and score - runner_up >= 0.025:
                    label = names[best]
        if merged and merged[-1]["chord"] == label:
            previous = merged[-1]
            previous["score"] = (previous["score"] * (previous["end"] - previous["start"])
                                  + score * (t1 - t0)) / (t1 - previous["start"])
            previous["end"] = t1
        else:
            merged.append({"start": t0, "end": t1, "chord": label, "score": score})
    return merged


def compact_waveform(y, bins=1400):
    width = max(1, math.ceil(len(y) / bins))
    count = math.ceil(len(y) / width)
    padded = np.pad(y, (0, count * width - len(y)), mode="edge").reshape(count, width)
    return {"time": np.minimum((np.arange(count) + 0.5) * width / SR, len(y) / SR),
            "min": padded.min(axis=1), "max": padded.max(axis=1)}


def analyze_audio(data: bytes, offset=0.0, seconds=120, window=1.0,
                  progress: Callable[[str], None] | None = None) -> dict:
    notify = progress or (lambda message: None)
    if not math.isfinite(window) or not 0.25 <= window <= 4:
        raise AudioError("和弦时间窗口须为 0.25–4 秒。")
    notify("读取所选片段，转换为单声道分析信号")
    y, meta = decode_audio(data, float(offset), int(seconds))
    duration = len(y) / SR
    notify("提取节奏、能量与色度特征")
    # Reuse one STFT instead of recalculating a spectrum for each feature.
    spectrum = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    magnitude = np.abs(spectrum)
    del spectrum
    power = magnitude ** 2
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    threshold = max(1e-5, float(rms.max()) * 0.03)
    active = rms > threshold
    chroma = librosa.feature.chroma_stft(S=power, sr=SR, hop_length=HOP, tuning=0.0)
    mel = librosa.feature.melspectrogram(S=power, sr=SR, n_mels=64)
    onset = librosa.onset.onset_strength(S=librosa.power_to_db(mel), sr=SR, hop_length=HOP)
    tempo, beats = 0.0, np.array([], dtype=int)
    if duration >= 4 and active.any():
        # 直接使用原始音频信号提取节拍，对流行乐更准
        bpm, beats = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP)
        tempo = float(np.atleast_1d(bpm)[0]) if len(beats) >= 2 else 0.0
    notify("匹配调式和声，合并连续片段")    
    mean = chroma[:, active].mean(axis=1) if active.any() else np.zeros(12)
    key = key_estimate(mean)
    chords = chord_segments(chroma, rms, duration, window, threshold)
    frequencies = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    centroid = (frequencies @ magnitude) / np.maximum(magnitude.sum(axis=0), 1e-10)
    notes = []
    if not active.any():
        notes.append("片段接近静音，无法可靠识别节拍、调式或和弦。")
    if key["label"] != "暂无法判断" and key["margin"] < 0.06:
        notes.append("前两种调式的匹配结果接近，建议结合旋律与结尾主音判断。")
    if duration < 10:
        notes.append("片段较短，节拍与调式估计可能不稳定。")
    if tempo == 0 and active.any():
        notes.append("未找到足够稳定的节拍；自由节奏或弱打击乐片段可能出现此情况。")
    if meta["offset"] > 0 or meta["offset"] + duration < meta["total_duration"] - 0.1:
        notes.append("本次结果仅描述所选片段，不代表整首歌曲。")
    # Bound chart payloads; do not retain the decoded audio or full spectrum.
    stride = max(1, math.ceil(chroma.shape[1] / 500))
    starts = np.arange(0, chroma.shape[1], stride)
    counts = np.minimum(stride, chroma.shape[1] - starts)
    reduced = np.add.reduceat(chroma, starts, axis=1) / counts
    energy_stride = max(1, math.ceil(len(rms) / 1000))
    return {**meta, "duration": duration, "tempo": tempo, "key": key, "chords": chords,
            "window": window, "notes": notes, "waveform": compact_waveform(y),
            "chroma": reduced.astype(np.float32), "chroma_times": (starts + counts / 2) * HOP / SR,
            "pitch_mean": mean, "energy_times": np.arange(0, len(rms), energy_stride) * HOP / SR,
            "energy_db": 20 * np.log10(np.maximum(rms[::energy_stride], 1e-9)),
            "rms_db": float(20 * np.log10(max(float(np.sqrt(np.mean(y ** 2))), 1e-9))),
            "centroid": float(centroid[active].mean()) if active.any() else 0.0,
            "peak": float(np.abs(y).max()), "beats": beats * HOP / SR}


def demo_audio() -> bytes:
    """Original synthetic C–G–Am–F example, 120 BPM; no downloaded assets."""
    rate = 22050
    length = 16
    y = np.zeros(rate * length, dtype=np.float32)
    for i, midi in enumerate(((60, 64, 67), (55, 59, 62), (57, 60, 64), (53, 57, 60)) * 2):
        t = np.arange(rate * 2, dtype=np.float32) / rate
        envelope = np.minimum(t / 0.035, 1) * np.minimum((2 - t) / 0.15, 1)
        chord = sum(np.sin(2 * np.pi * (440 * 2 ** ((m - 69) / 12)) * t) for m in midi)
        y[i * rate * 2:(i + 1) * rate * 2] = chord * envelope * 0.15
    for beat in np.arange(0, length, 0.5):
        t = np.arange(int(rate * 0.08)) / rate
        click = 0.12 * np.sin(2 * np.pi * 1100 * t) * np.exp(-t * 80)
        begin = int(beat * rate)
        y[begin:begin + len(click)] += click
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()
