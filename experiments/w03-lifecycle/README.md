# W03 请求生命周期教学模拟

本目录不导入 vLLM、不调用 GPU。固定三个合成请求，输出调度步骤、计数、块容量和取消后的状态；这些结果不含毫秒或真实引擎吞吐。

## 运行

在学习仓库根目录：

```powershell
py -3.11 experiments/w03-lifecycle/simulate.py
py -3.11 -m unittest discover -s experiments/w03-lifecycle -p test_simulate.py -v
```

Python 3.11 标准库即可。结果写入 `data/`；`--output` 可以指定另一结果目录。无随机数，JSON 与 CSV 可逐字节复现。取消在 step 2 的准入和执行之前发生。`before` 是到达/取消处理后、计划前的快照；`after` 是执行并释放完成请求后的快照；`peak_blocks` 是计划完成、执行释放之前的占用。

## 模型规则

- A=(prompt 4, output 2, arrival 0)，B=(2,3,0)，C=(6,1,1)。arrival 为离散步编号，未给各步赋时长。
- running 优先，然后从 waiting 按先来先服务准入。计划时已在 running 的请求占用活动名额，不能预借步末释放的名额。
- 单请求待计算位置为 prompt + emitted − computed。部分 prefill 不输出；追上本步开始时已知序列后输出一个 token。
- KV 为单组、无共享；所需块为 ceil((computed+n)/block_size)。达到输出上限后释放该请求所有块。
- 不执行抢占。若有积压且一步无法安排任何请求，结果为 `BLOCKED_NO_PREEMPTION`，保留失败状态供分析，不能视为完成。

真实 vLLM 的分支差异见[教材](../../lessons/week03/README.md)第 6 节。尤其注意真实 scheduler 可能提前推进 computed，并不与此同步模型使用相同的更新时间点。

## 结果与校验

- `baseline.json`：四步手算对照；`budget2.json`、`serial.json`、`wide.json`：单纯改变预算或活动上限。
- `cancel.json`：先产生 B 的第一个输出，再取消；它改变交付工作量。
- `blocked.json`：容量不足且没有抢占策略的反例；`summary.csv`：配置对比摘要。
- 8 个单元测试包括手算 oracle、首 token/部分 prefill、最后 token 未 forward、取消、容量失败、串行顺序及 192 种参数组合的不变量（8 预算 × 3 活动上限 × 4 容量 × 2 取消选项 = **192**）。

## 重建教材与图表

已有 `.venv` 可直接使用。新机器先建立环境，再安装锁定依赖：

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install -r experiments/w03-lifecycle/requirements.txt
py -3.11 lessons/week03/source_evidence.py
.venv/Scripts/python.exe lessons/week03/build.py
node lessons/week03/check_interactive.cjs
```

source_evidence 需要完整锁文件和原源码目录；离线 HTML 的阅读、交互本身不需要它们。构建将源码证据表与生成数据汇总进 Markdown，并内嵌图片、CSS、JavaScript，生成单文件 `index.html`。保留仓库目录才可打开附加数据、代码和源码链接。图表提供 PNG（300 DPI）与 PDF；两张结构示意图使用可缩放 SVG。
