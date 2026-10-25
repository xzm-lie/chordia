from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis import AudioError, NOTES, analyze_audio, demo_audio
from reports import ReportError, export_markdown, generate_report, local_report, stamp, summary

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Chordia 和弦之境 · 音乐分析工作台", page_icon="◉", layout="wide")
st.markdown("<style>" + (ROOT / "style.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)


def config(name, default=""):
    value = os.environ.get(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except FileNotFoundError:
        return default


def chart_style(fig, height=250):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9bafbf", family="Arial, Microsoft YaHei, sans-serif", size=11),
        margin=dict(l=12, r=12, t=12, b=28), showlegend=False,
        hoverlabel=dict(bgcolor="#162631", font_color="#eaf7f5"),
    )
    fig.update_xaxes(gridcolor="#21303d", zeroline=False, title_font_size=11)
    fig.update_yaxes(gridcolor="#21303d", zeroline=False, title_font_size=11)
    return fig


def show_chart(fig, key):
    st.plotly_chart(fig, width="stretch", key=key,
                    config={"displaylogo": False, "scrollZoom": False,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def waveform_chart(result):
    wave = result["waveform"]
    x = wave["time"] + result["offset"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=wave["min"], mode="lines", line=dict(color="#78e5cf", width=.6),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=wave["max"], mode="lines", line=dict(color="#78e5cf", width=.6),
                             fill="tonexty", fillcolor="rgba(120,229,207,.32)",
                             hovertemplate="%{x:.2f} 秒<br>峰值 %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title="原音频位置 / 秒", range=[result["offset"], result["offset"] + result["duration"]])
    fig.update_yaxes(title="振幅", range=[-max(result["peak"] * 1.12, .05), max(result["peak"] * 1.12, .05)])
    return chart_style(fig, 240)


def timeline_chart(result):
    fig = go.Figure()
    grouped = {}
    for chord in result["chords"]:
        grouped.setdefault(chord["chord"], []).append(chord)
    for label, chunks in grouped.items():
        root = label.rstrip("m")
        color = "#334150" if label == "N" else f"hsl({155 + NOTES.index(root) * 19},43%,61%)"
        fig.add_trace(go.Bar(
            x=[c["end"] - c["start"] for c in chunks], y=["和弦"] * len(chunks),
            base=[c["start"] + result["offset"] for c in chunks], orientation="h", width=.65,
            marker=dict(color=color, line=dict(color="#0e161f", width=1)), name=label,
            text=[label] * len(chunks), textposition="inside", insidetextanchor="middle",
            textfont=dict(color="#081719", size=12),
            customdata=[[c["start"] + result["offset"], c["end"] + result["offset"], c["score"]] for c in chunks],
            hovertemplate=label + "<br>%{customdata[0]:.2f}–%{customdata[1]:.2f} 秒"
                          "<br>模板相似度 %{customdata[2]:.2f}<extra></extra>",
        ))
    fig.update_layout(barmode="overlay", uniformtext=dict(minsize=10, mode="hide"))
    fig.update_xaxes(title="原音频位置 / 秒", range=[result["offset"], result["offset"] + result["duration"]])
    fig.update_yaxes(visible=False)
    return chart_style(fig, 150)


def chroma_chart(result):
    fig = go.Figure(go.Heatmap(
        z=result["chroma"], x=result["chroma_times"] + result["offset"], y=list(NOTES),
        colorscale=[[0, "#101a28"], [.3, "#214455"], [.65, "#50a99d"], [1, "#c8f8bd"]],
        zmin=0, zmax=1, colorbar=dict(title="强度", thickness=9),
        hovertemplate="%{x:.2f} 秒 · %{y}<br>相对强度 %{z:.2f}<extra></extra>",
    ))
    fig.update_xaxes(title="原音频位置 / 秒")
    fig.update_yaxes(title="音高类别")
    return chart_style(fig, 340)


def render_metrics(result):
    recognized = sum(c["end"] - c["start"] for c in result["chords"] if c["chord"] != "N")
    cards = [
        ("分析片段", stamp(result["duration"]), f"原音频 {stamp(result['total_duration'])}"),
        ("节拍估计", f"{result['tempo']:.1f}" if result["tempo"] else "—", "BPM · 可能有半速 / 倍速误差"),
        ("调式候选", result["key"]["label"], "Krumhansl–Schmuckler 轮廓匹配"),
        ("已匹配和声", f"{recognized / result['duration']:.0%}", "匹配到三和弦的时长占比，非准确率"),
    ]
    markup = '<div class="metric-grid">'
    for label, value, caption in cards:
        markup += (f'<div class="metric"><div class="metric-label">{html.escape(label)}</div>'
                   f'<div class="metric-value">{html.escape(value)}</div>'
                   f'<div class="metric-caption">{html.escape(caption)}</div></div>')
    st.markdown(markup + "</div>", unsafe_allow_html=True)


with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-icon">◉</div><div><div class="brand-name">Chordia</div>'
                '<div class="brand-sub">和弦之境 / MUSIC LAB</div></div></div>', unsafe_allow_html=True)
    st.caption("01 / 导入声音")
    uploaded = st.file_uploader("选择一段音乐", type=["mp3", "wav", "flac", "ogg"],
                                help="支持 MP3、WAV、FLAC、OGG，最大 100 MB。")
    st.caption("02 / 选择分析范围")
    offset = st.number_input("从第几秒开始", min_value=0.0, value=0.0, step=5.0)
    seconds = st.select_slider("最多分析", options=[30, 60, 120, 180, 300], value=120,
                                format_func=lambda n: f"{n} 秒")
    window = st.select_slider("和弦时间窗口", options=[0.5, 1.0, 2.0], value=1.0,
                              format_func=lambda n: f"{n:g} 秒",
                              help="较短窗口捕捉快速变化；较长窗口更平稳。")
    run = st.button("开始分析  →", type="primary", width="stretch", disabled=uploaded is None)
    demo = st.button("试听并分析示例", width="stretch")
    st.caption("示例为本地合成的 16 秒 C–G–Am–F 和声，标称 120 BPM；从头分析。")
    
    with st.expander("AI 分析设置 · 可选", expanded=False):
        saved_key = config("DEEPSEEK_API_KEY")
        entered_key = st.text_input("DeepSeek API 密钥", type="password", key="api_key_input",
                                    help="仅保留在当前会话；用于测试阶段。正式合规生成请前往 WorkBuddy 平台。")
        api_key = entered_key.strip() or saved_key.strip()
        model = st.text_input("模型名称", value=config("DEEPSEEK_MODEL", "deepseek-chat"))
        st.caption("已配置密钥" if api_key else "未配置密钥 · 本地分析不受影响")
        st.caption("仅在点击生成乐评后，向 DeepSeek 发送声学摘要。正式合规生成请前往官方 WorkBuddy 平台。")
    st.divider()
    st.caption("声学分析在运行此应用的电脑 / 服务器完成。首次分析可能稍慢。")

st.markdown('<div class="masthead"><span>CHORDIA / 音乐分析工作台</span>'
            '<span class="status-pill">● 本地声学分析</span></div>', unsafe_allow_html=True)

if run or demo:
    data = demo_audio() if demo else uploaded.getvalue()
    name = "C–G–Am–F · 合成示例.wav" if demo else uploaded.name
    start = 0.0 if demo else float(offset)
    identity = (hashlib.sha256(data).hexdigest(), start, seconds, window)
    previous = st.session_state.get("bundle")
    if previous and previous["identity"] == identity:
        previous["name"] = name
        st.success("已读取当前会话中的分析结果，无需重复计算。")
    else:
        with st.status("正在分析声音…", expanded=True) as status:
            started = time.perf_counter()
            try:
                result = analyze_audio(data, start, seconds, window, progress=st.write)
                st.session_state.bundle = {
                    "result": result, "data": data, "name": name, "identity": identity,
                    "elapsed": time.perf_counter() - started, "report": "", "is_demo": demo,
                }
                status.update(label="分析完成", state="complete", expanded=False)
            except AudioError as exc:
                status.update(label="暂时无法分析这段音频", state="error", expanded=True)
                st.error(str(exc))
            except Exception as exc:
                status.update(label="分析未完成", state="error", expanded=True)
                st.error(f"处理时出现 {type(exc).__name__}。请尝试较短片段或标准 WAV 文件；已有结果会保留。")

bundle = st.session_state.get("bundle")
if not bundle:
    st.markdown('<div class="hero"><div class="orb"></div><div class="eyebrow">LISTEN. EXPLORE. CREATE.</div>'
                '<h1>听见声音背后的结构。</h1><p>从一个节拍，到一段和声。把音乐里的细节，变成可探索的线索。'
                '<br>上传音频，开启你的下一次音乐发现。</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="steps">'
                '<div class="step"><div class="step-number">01 / RHYTHM</div><div class="step-title">找到节奏的脉搏</div>'
                '<div class="step-text">观察波形与能量起伏，了解片段的节拍和动态变化。</div></div>'
                '<div class="step"><div class="step-number">02 / HARMONY</div><div class="step-title">看见和声的走向</div>'
                '<div class="step-text">沿着时间轴探索和弦候选，用音高热力图辅助听辨。</div></div>'
                '<div class="step"><div class="step-number">03 / INSIGHT</div><div class="step-title">把分析变成灵感</div>'
                '<div class="step-text">阅读声学摘要，按需生成 AI 解读，导出你的分析笔记。</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="empty-note"><strong>还没准备好音乐？</strong><br>'
                '点击左侧「试听并分析示例」，用一段原创合成和声体验完整工作流，无需 API 密钥。</div>', unsafe_allow_html=True)
else:
    result = bundle["result"]
    st.markdown('<div class="hero compact"><div class="orb"></div><div class="eyebrow">YOUR SOUND, IN FOCUS</div>'
                '<h1>让每个细节被听见。</h1><p>探索节拍、和声与音高分布，让灵感有据可循。</p></div>', unsafe_allow_html=True)
    if uploaded is not None and (hashlib.sha256(uploaded.getvalue()).hexdigest() != bundle["identity"][0]
                                or (float(offset), seconds, window) != bundle["identity"][1:]):
        st.info("上传文件或分析设置已变更。下方仍是上次结果，点击「开始分析」更新。")
    elif (seconds, window) != bundle["identity"][2:]:
        st.info("分析设置已变更。下方仍显示上次结果。")
    st.markdown(f'<div class="filename">{html.escape(bundle["name"])}</div>'
                f'<div class="meta">{result["source_sr"]:,} Hz · {result["channels"]} 声道 · '
                f'{stamp(result["offset"])}–{stamp(result["offset"] + result["duration"])} · '
                f'计算用时 {bundle["elapsed"]:.2f} 秒</div>', unsafe_allow_html=True)
    suffix = Path(bundle["name"]).suffix.lower()
    mime = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg"}.get(suffix, "audio/wav")
    st.audio(bundle["data"], format=mime, start_time=int(result["offset"]))
    render_metrics(result)

    overview, harmony, report_tab, export_tab = st.tabs(["声学概览", "和声探索", "分析笔记", "导出结果"])
    with overview:
        with st.container(border=True):
            st.subheader("声音的轮廓")
            st.caption("拖动框选以放大，双击还原；波形按区间保留峰值。")
            show_chart(waveform_chart(result), "waveform")
        left, right = st.columns([1.45, 1], gap="large")
        with left, st.container(border=True):
            st.subheader("能量起伏")
            fig = go.Figure(go.Scatter(x=result["energy_times"] + result["offset"],
                                      y=result["energy_db"], mode="lines", line=dict(color="#b7c6fa", width=1.6),
                                      hovertemplate="%{x:.2f} 秒<br>%{y:.1f} dBFS<extra></extra>"))
            fig.update_xaxes(title="原音频位置 / 秒")
            fig.update_yaxes(title="RMS / dBFS")
            show_chart(chart_style(fig, 235), "energy")
        with right, st.container(border=True):
            st.subheader("片段特征")
            st.metric("平均 RMS 电平", f"{result['rms_db']:.1f} dBFS")
            st.metric("活跃帧频谱重心", f"{result['centroid']:,.0f} Hz")
            st.caption("频谱重心反映频谱分布；RMS 电平不是 LUFS，也不是混音质量评分。")
        if result["notes"]:
            with st.expander("如何理解这次结果", expanded=True):
                for note in result["notes"]:
                    st.write("• " + note)

    with harmony:
        with st.container(border=True):
            st.subheader("和弦时间轴")
            st.caption("仅匹配大、小三和弦。N = 静音或无法明确匹配；相似度不是准确率。")
            show_chart(timeline_chart(result), "chord_timeline")
            chips = "".join(f'<span class="chord{" unknown" if c["chord"] == "N" else ""}">{c["chord"]}</span>'
                            for c in result["chords"][:36])
            st.markdown('<div class="chords">' + chips + "</div>", unsafe_allow_html=True)
            if len(result["chords"]) > 36:
                st.caption("标签仅展示前 36 段；时间轴与导出包含完整结果。")
        with st.container(border=True):
            st.subheader("音高色度图")
            st.caption("颜色越亮，对应音高类别的相对能量越强；所有八度合并显示。")
            show_chart(chroma_chart(result), "chroma")
        left, right = st.columns([1.3, 1], gap="large")
        with left, st.container(border=True):
            st.subheader("音高分布")
            fig = go.Figure(go.Bar(x=list(NOTES), y=result["pitch_mean"],
                                  marker_color=["#78e5cf" if "#" not in note else "#537e88" for note in NOTES],
                                  hovertemplate="%{x}<br>相对强度 %{y:.3f}<extra></extra>"))
            show_chart(chart_style(fig, 230), "pitch_profile")
        with right, st.container(border=True):
            st.subheader("调式候选")
            for candidate in result["key"]["alternatives"]:
                st.write(f"**{candidate['label']}** · 相关系数 {candidate['score']:.3f}")
            if not result["key"]["alternatives"]:
                st.write("当前片段没有足够的调性信息。")
            st.caption("这是音高分布与调式轮廓的相关程度，不是概率。相关系数接近时，需结合旋律和终止式听辨。")

    with report_tab:
        left, right = st.columns([1, 1], gap="large")
        with left, st.container(border=True):
            st.subheader("声学笔记")
            st.caption("本地生成 · 不调用 AI")
            st.markdown(local_report(result))
            
            # ====== 新增的一键复制功能开始 ======
            st.divider()
            st.subheader("导出分析提示词")
            st.caption("👇 请点击代码块右上角的复制图标，前往官方 WorkBuddy 平台粘贴发送。")
            
            # 生成干净的 Prompt
            clean_chords = " -> ".join(c["chord"] for c in result["chords"][:60] if c["chord"] != "N")
            if not clean_chords:
                clean_chords = " -> ".join(c["chord"] for c in result["chords"][:30])
                
            copy_text = f"""你是一位学院派的硬核音乐理论分析师。请基于以下声学测量结果进行硬核乐理分析。

【声学数据】
分析时长：约 {result['duration']:.1f} 秒
节奏 (BPM)：{result['tempo']:.1f}
主调与调式：{result['key']['label']}
和弦走向（节选）：{clean_chords}

【输出要求】
请直接输出以下三个部分的内容（字数300-400字）：
1. 【调式色彩分析】：结合给定的主调，分析其带来的音响色彩和情绪倾向。
2. 【和声走向剖析】：分析给定的和弦序列，指出其中包含了哪些经典的级数进行（如I-V-vi-IV），并解释这种走向的张力与解决感。如果出现非自然音阶和弦，请指出其可能带来的离调感。
3. 【编曲建议】：基于当前的BPM和和声密度，给出一条非常专业的编曲或配器建议。

【注意】：严禁使用“数据不足”、“无法判断”等推托之词；不要解释“N”是什么意思；直接把给定的和弦当作真实存在的进行剖析；直接输出分析正文，不要任何开场白。"""
            
            st.code(copy_text, language="markdown")
            # ====== 新增的一键复制功能结束 ======

        with right, st.container(border=True):
            st.subheader("AI 辅助解读")
            st.caption("仅根据声学摘要解读，不等同于模型实际听过音频。")
            if st.button("重新生成解读" if bundle["report"] else "生成 AI 解读  ↗",
                         disabled=not api_key or not model.strip(), width="stretch"):
                with st.spinner("正在生成解读，通常需要数秒…"):
                    try:
                        bundle["report"] = generate_report(result, api_key, model)
                    except ReportError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(f"AI 解读暂未完成（{type(exc).__name__}），已有结果已保留。")
            if bundle["report"]:
                st.markdown(bundle["report"])
            elif not api_key:
                st.info("为确保赛事平台合规，请复制左侧的声学数据（BPM、调式、和弦），前往官方 **WorkBuddy 平台** 完成交互式生成。\n\n*(在左侧配置 API 密钥可用于本地开发测试)*")
            else:
                st.info("准备好了。点击上方按钮，生成节奏、和声与创作建议。")

    with export_tab:
        st.subheader("带走这次音乐发现")
        st.caption("导出完整的和弦时间戳、结构化数据和分析笔记。")
        frame = pd.DataFrame(summary(result)["chord_segments"])
        frame.columns = ["开始时间(秒)", "结束时间(秒)", "和弦", "模板相似度"]
        columns = st.columns(3)
        with columns[0]:
            st.download_button("↓ 分析报告 · Markdown", export_markdown(result, bundle["name"], bundle["report"]),
                               file_name="chordia-report.md", mime="text/markdown", width="stretch", on_click="ignore")
        with columns[1]:
            st.download_button("↓ 和弦时间轴 · CSV", frame.to_csv(index=False).encode("utf-8-sig"),
                               file_name="chordia-chords.csv", mime="text/csv", width="stretch", on_click="ignore")
        with columns[2]:
            st.download_button("↓ 结构化数据 · JSON", json.dumps(summary(result), ensure_ascii=False, indent=2),
                               file_name="chordia-analysis.json", mime="application/json", width="stretch", on_click="ignore")
        st.dataframe(frame, hide_index=True, width="stretch", height=360)

st.markdown('<div class="footer"><span>Chordia 和弦之境 · 把声音变成可探索的线索</span>'
            '<span>声学估计辅助听辨 · 请以实际听感为准</span></div>', unsafe_allow_html=True)