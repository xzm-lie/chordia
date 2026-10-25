# CHORDIA 和弦之境 · 音乐分析工作台

基于你提供的 Streamlit 脚本重新整理的可运行版本。保留音频上传、节拍估计、大小调识别、和弦分析和 DeepSeek 乐评，增加交互图表、状态保留、导出和无密钥示例。

## 开始使用

推荐 Python 3.11 或 3.12。在解压后的 `sonora` 文件夹中打开终端：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

无需激活虚拟环境，也不需要修改 PowerShell 执行策略。macOS / Linux 将 `.\.venv\Scripts\python.exe` 换成 `.venv/bin/python`。

首次安装需要网络。首次运行音频算法时，Numba 可能需要编译，通常比后续分析慢。启动后点击“试听并分析示例”，即可用原创合成音频体验所有本地功能。

如果你原来的 Python 环境已经安装这些库，也可先更新依赖，然后运行：

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

请保留项目目录结构，不能只复制 `app.py`：

```text
sonora/
  app.py                  界面、图表、交互和会话状态
  analysis.py             音频解码、特征提取、调式与和弦匹配
  reports.py              本地笔记、导出、可选 AI 请求
  style.css               深色界面和窄屏样式
  requirements.txt        运行依赖
  .streamlit/
    config.toml           主题、上传体积限制
    secrets.toml.example  密钥配置示例（不含真实密钥）
  tests/                  算法和页面交互测试
```

## AI 配置（可选）

原文件中包含明文密钥。交付版没有沿用或保存它，请在服务商控制台撤销旧密钥并创建新密钥。

任选一种方式：

1. 在侧栏“AI 乐评设置”中输入密钥，仅保留在当前会话。
2. 将 `.streamlit/secrets.toml.example` 复制为 `.streamlit/secrets.toml`，填写新密钥。不要上传或提交该文件。
3. 启动前设置环境变量：

```powershell
$env:DEEPSEEK_API_KEY = "在这里填写你的新密钥"
$env:DEEPSEEK_MODEL = "deepseek-flash"
python -m streamlit run app.py
```

读取优先级：侧栏输入 > 环境变量 > secrets.toml。模型名称可在侧栏修改；默认值依照检查时的 [DeepSeek 官方接口文档](https://api-docs.deepseek.com/) 设置。接口地址固定为 `https://api.deepseek.com`。

本地分析不会调用 AI。只有点击“生成 AI 解读”才发起一次请求，可能产生服务商费用。发送内容仅为节拍、调式候选、声学指标和最多 80 个和弦片段的摘要，不发送原音频、文件名或图表。请求设置 60 秒超时，不自动重试；失败不会清除本地结果或已有报告。

真实付费 API 调用需要你自己的有效密钥；交付测试使用模拟响应，不消耗账户额度。

## 主要变化

- **界面**：深色工作台、薄荷绿强调色、概览卡片、四个内容页签、统一图表样式和窄屏适配。
- **交互**：音频播放器、可缩放波形、能量曲线、色度热力图、音高分布、带原音频时间戳的和弦轨道。
- **性能**：默认分析 120 秒，可选 30–300 秒；支持指定起点。分块读取和流式重采样，避免先加载整首高采样率多声道音频。多项特征复用一次 STFT。
- **内存**：只保留当前分析结果；不把完整频谱或解码后的音频放入会话。波形最多 1400 个区间、热力图最多 500 列、能量图最多 1000 点；完整和弦时间轴仍可导出。原文件字节会保留用于试听。
- **缓存与状态**：用文件内容哈希与参数识别当前结果；普通界面交互不触发分析，同一结果可复用；更换文件或参数时提示结果尚未更新。刷新浏览器或断开会话可能清除状态，因此需要保留时请导出。
- **算法修正**：处理静音、平坦色度和无效输入；使用归一化和弦模板及匹配差距阈值；保留不足一个窗口的尾段；合并和弦时保留起止时间。
- **可靠性**：上传体积上限 50 MB，分析时长上限 300 秒；校验解码、片段范围及音频采样值；不生成临时音频文件，因而没有异常退出后的临时文件残留。
- **导出**：Markdown 报告、带 UTF-8 BOM 的 CSV（便于 Excel 打开）、结构化 JSON。导出不重新计算音频或再次调用 AI。

会话状态行为参考 [Streamlit 文档](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state)，下载交互参考 [下载按钮文档](https://docs.streamlit.io/develop/api-reference/widgets/st.download_button)。

## 识别边界

本工具是轻量声学探索工具，不是专业自动扒谱软件。分析前将音频转换成单声道，调音基准为十二平均律、A4=440 Hz。

- 和弦仅覆盖 12 个大三和弦及 12 个小三和弦。七和弦、转位、复杂配器、噪声及弱和声可能被简化或误判。`N` 表示静音或无法明确匹配。
- 调式依赖音高分布，关系大小调可能接近；短片段和转调作品未必有唯一答案。
- BPM 可能出现半速或倍速估计。自由节奏和缺乏明显瞬态的音乐可能无法可靠检出。
- 模板相似度、轮廓相关系数和已匹配和声时长占比都不是识别准确率或置信概率。
- RMS 是电平，频谱重心是频谱分布指标；两者不能直接决定曲风、情绪或混音质量。AI 也没有直接听到音频。
- 所选片段之外的内容没有参与分析。长文件请按不同起点分段分析。
- 本地运行时声学分析留在你的电脑；部署到服务器时，上传文件会传到该服务器，在当前会话中处理。不要将“本地算法”理解为浏览器端运算。

## 检查与测试

在项目目录执行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

覆盖调式轮廓旋转、合成和弦、静音、短片段尾部、时间轴连续性、多声道重采样、无效文件、导出数据、AI 模拟成功/空响应/超时、界面状态与重复分析缓存。测试不发送真实 API 请求。

性能随硬件、音频长度和首次编译情况变化；没有声称固定处理速度或对真实歌曲的识别准确率。
