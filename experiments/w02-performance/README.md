# W02 · 推理性能分析实验

[图文讲义](../../lessons/week02/README.md) · [离线交互版](../../lessons/week02/index.html) · [本周实验协议](protocol.md)

本实验提供 **公式估算、构造流式 trace 和单服务器 FCFS 模拟**。没有调用真实 LLM，也没有测量 CPU/GPU、SSD、网络或 PCIe 的延迟/带宽。

## 快速运行

从学习仓库根目录执行，复用第一周 .venv：

```powershell
.venv\Scripts\python.exe experiments/w02-performance/test_models.py
.venv\Scripts\python.exe experiments/w02-performance/run.py
.venv\Scripts\python.exe experiments/w02-performance/estimate.py --requests 8 --context 8192 --bandwidth-gbs 25 --chunk-mib 4
.venv\Scripts\python.exe lessons/week02/build.py
node lessons/week02/check_interactive.cjs
```

最后一条只验证交互逻辑；打开 HTML 无需 Node。新的 Python 3.11+ 环境可直接运行前三条对应脚本，它们只依赖标准库；画图需要下列版本。没有 uv 时可用普通 Python venv 与 pip 创建环境。

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r experiments/w02-performance/requirements.txt
```

版本来自本机已验证环境：Python 3.11.0、Matplotlib 3.11.2、NumPy 2.4.6、Markdown 3.11。绘图使用 Windows 微软雅黑；其他系统需要 Noto Sans CJK SC 中文字体。build.py 复用仓库内第一周基础样式，生成后的 HTML 将所有样式、脚本和图片内嵌，单文件可离线使用。

## 四份可读代码

| 文件 | 阅读顺序与职责 |
| --- | --- |
| models.py | 先读 kv_bytes → transfer → gemm，再读 request_metrics → summarize → fcfs |
| estimate.py | 命令行参数与单位；用 --help 查看完整参数 |
| run.py | 构造数据、固定随机种子、扫描参数并导出 |
| test_models.py | 18 项手算、边界和语义检查；不是性能 benchmark |

估算默认配置对应：1 GiB KV，在假设 25 GB/s 路径下纯载荷 42.94967296 ms；4 MiB 一块、每块 40 μs 串行开销时为 53.18967296 ms。带宽单位是 **十进制 GB/s**，不是 Gbit/s。CLI 对 0 带宽、负数、非整数形状和非有限数拒绝计算。

## 数据与复现范围

- gemm.csv：6 组线性层形状的理想运算/读写下界。
- batch.csv：3 种上下文 × 7 种活动 batch 的简化模型，固定假想算力和带宽，无内存容量限制。
- transfer.csv：5 种上下文 × 2 种请求数 × 3 种带宽 × 3 种 chunk，合计 90 组。
- request_trace.json / metrics.json：5 个构造请求，包含客户端等待、停顿、单 token 与失败。统计窗口 0–500 ms，包含排空与失败；成功 token 共 12。
- queue_summary.csv：6 个利用率 × 5 个种子，30 轮模拟；每轮先处理 2000 请求，再统计后续 10000 请求。
- queue_sample.csv：ρ=0.9、首个 seed 的观测区前 200 个请求。只用于手算递推，完整分位数需要按 run.py 重建整轮。
- metadata.json：明确证据类型、假设、Python 版本、分位数算法、种子与代码 SHA-256。

指标使用 nearest rank 分位数，空样本为 null；TPOT 对单 token 为 null。总体 TPOT 按请求等权，ITL 按观测间隔等权。SLO 的单 token 处理在输出 JSON 中写明。输出吞吐与请求吞吐使用相同窗口，失败不进入成功分子，但不会从窗口中消失。

## 从“跑通”走到自己的结论

先独立手算，再运行默认脚本；把一个改动放在独立分支或保存参数副本，重建输出并填写 protocol.md 的学习记录。默认数据是助手准备的例子，不能记为自己的学习验收。

未来接入真实服务 trace 时，要先映射 scheduled、send、输出流块、token 计数、结束、错误这几个字段。不要把未知 token 数默认填成 1；不要把整个响应合并成一个时间戳后还声称测到了逐 token ITL。

GPU 尚未就绪时，完成模拟协议和计算验收即可；真实吞吐、TTFT、ITL、内存峰值与资源成本字段保留“待测”。
