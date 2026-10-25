"""Report formatting and an optional, explicitly triggered DeepSeek request."""
from __future__ import annotations

import json


def stamp(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def summary(result: dict, limit: int | None = None) -> dict:
    chords = result["chords"] if limit is None else result["chords"][:limit]
    return {
        "analysis_range_seconds": [result["offset"], result["offset"] + result["duration"]],
        "source_duration_seconds": result["total_duration"],
        "tempo_bpm_estimate": result["tempo"] or None,
        "key_estimate": result["key"],
        "rms_dbfs": result["rms_db"],
        "spectral_centroid_hz": result["centroid"],
        "chord_window_seconds": result["window"],
        "chord_segments": [{"start": round(c["start"] + result["offset"], 3),
                            "end": round(c["end"] + result["offset"], 3),
                            "chord": c["chord"], "cosine_similarity": round(c["score"], 4)}
                           for c in chords],
        "segment_count": len(result["chords"]),
        "segments_truncated": len(chords) < len(result["chords"]),
        "notes": result["notes"],
        "limitations": "Only major/minor triads; N means silence or uncertain harmony. "
                       "Template scores are not probabilities. Tempo may be half/double time. "
                       "No direct evidence of lyrics, instruments, genre, or mixing quality.",
    }


def local_report(result: dict) -> str:
    bpm = f"约 {result['tempo']:.1f} BPM" if result["tempo"] else "暂未可靠检出"
    progression = " → ".join(c["chord"] for c in result["chords"][:24]) or "暂无"
    if len(result["chords"]) > 24:
        progression += " → …"
    notes = "\n".join(f"- {note}" for note in result["notes"])
    return (
        f"### 片段概览\n\n"
        f"本次分析覆盖 **{stamp(result['offset'])}–{stamp(result['offset'] + result['duration'])}**，"
        f"节拍估计为 **{bpm}**，调式候选为 **{result['key']['label']}**。\n\n"
        f"### 和声线索\n\n{progression}\n\n"
        "这是基于色度特征的大、小三和弦模板匹配。N 表示静音或无法明确匹配，"
        "七和弦、转位和复杂配器可能被简化或误判。\n\n"
        "### 下一步怎么听\n\n"
        "跟随波形和和弦时间轴回听变化点；尝试不同时间窗口，核对和弦切换位置。"
        "若节拍听起来不对，可对比估计值的一半或两倍。\n\n"
        "均方根电平与频谱重心用于描述信号强度和频谱分布，不能直接评价混音质量。"
        "仅凭这些特征，也无法确认人声、歌词、具体乐器或曲风。\n\n" + notes
    )


class ReportError(RuntimeError):
    pass


def generate_report(result: dict, api_key: str, model: str) -> str:
    from openai import OpenAI, AuthenticationError, RateLimitError, APITimeoutError, APIConnectionError, APIStatusError

    if not api_key.strip():
        raise ReportError("请先在侧栏填写 API 密钥，或配置 DEEPSEEK_API_KEY。")
    if not model.strip():
        raise ReportError("请填写可用的模型名称。")
        
    # 1. 提取最关键的数据，过滤掉无意义的 "N"
    chords_str = " -> ".join(c["chord"] for c in result["chords"][:60] if c["chord"] != "N")
    if not chords_str:
        chords_str = " -> ".join(c["chord"] for c in result["chords"][:30])
        
    analysis_data = (
        f"分析时长：约 {result['duration']:.1f} 秒\n"
        f"节奏 (BPM)：{result['tempo']:.1f}\n"
        f"主调与调式：{result['key']['label']}\n"
        f"和弦走向（节选）：{chords_str}\n"
    )
    
    try:
        with OpenAI(api_key=api_key.strip(), base_url="https://api.deepseek.com",
                    timeout=60.0, max_retries=0) as client:
            response = client.chat.completions.create(
                model=model.strip(), max_tokens=1600,
                messages=[
                    {"role": "system", "content": 
                     "你是一位学院派的硬核音乐理论分析师。你不需要抒情，不需要写散文，只需要进行专业的乐理分析。\n"
                     "请直接输出以下三个部分的内容（字数300-400字）：\n"
                     "1. 【调式色彩分析】：结合给定的主调（如B大调），分析其带来的音响色彩和情绪倾向。\n"
                     "2. 【和声走向剖析】：分析给定的和弦序列，指出其中包含了哪些经典的级数进行（如I-V-vi-IV），并解释这种走向的张力与解决感。如果出现非自然音阶和弦，请指出其可能带来的离调感。\n"
                     "3. 【编曲建议】：基于当前的BPM和和声密度，给出一条非常专业的编曲或配器建议。\n"
                     "【注意】：严禁使用“数据不足”、“无法判断”等推托之词；不要解释“N”是什么意思；直接把给定的和弦当作真实存在的进行剖析；直接输出分析正文，不要任何开场白。"},
                    {"role": "user", "content": "请基于以下声学测量结果进行硬核乐理分析：\n" + analysis_data},
                ],
            )
            
        # 兼容性抓取：优先获取正文，如果正文为空则尝试获取思考内容
        message = response.choices[0].message if response.choices else None
        content = None
        if message:
            content = message.content
            if not content and hasattr(message, 'reasoning_content'):
                content = message.reasoning_content

        if not content or not str(content).strip():
            raise ReportError("模型没有返回正文，请稍后重试或检查模型名称。")
        return str(content).strip()
        
    except AuthenticationError as exc:
        raise ReportError("密钥验证失败，请检查密钥是否正确、是否已失效。") from exc
    except RateLimitError as exc:
        raise ReportError("请求受限，请检查账户额度或稍后重试。") from exc
    except (APITimeoutError, APIConnectionError) as exc:
        raise ReportError("服务暂时无法连接或请求超时。本地分析已保留，可稍后重试。") from exc
    except APIStatusError as exc:
        raise ReportError(f"服务返回 HTTP {exc.status_code}，请检查模型设置或稍后重试。") from exc


def export_markdown(result: dict, name: str, ai_report: str = "") -> str:
    title = name.replace("\n", " ").replace("\r", " ")
    return ("# Chordia · 音乐分析报告\n\n文件：" + json.dumps(title, ensure_ascii=False) + "\n\n"
            + local_report(result)
            + ("\n\n---\n\n## AI 辅助解读\n\n" + ai_report if ai_report else "")
            + "\n\n---\n\n本报告由声学特征和三和弦模板估计生成，不是人工扒谱。\n")