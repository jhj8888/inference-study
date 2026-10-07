# 第一周：Attention 与 KVCache 实验

教学材料见 [完整讲义](../../lessons/week01/README.md) 与 [离线交互版](../../lessons/week01/index.html)。

## 实验目的与范围

比较同一模型、相同输入下完整前缀计算与增量 KVCache 的 logits。覆盖 MHA/GQA/MQA、两层因果 decoder、RoPE、单 token 与 chunk 输入、贪心生成；通过四个故意错误检查掩码、位置、权重和前缀变化。

这是 CPU float64 正确性实验。随机初始化的小模型不能用于语言质量评价；没有记录延迟，不用它推断 GPU 加速比。PyTorch concat 和显式 repeat_interleave 会产生真实服务实现通常避免的额外开销。

## 环境与运行

当前已验证 Windows、Python 3.11.0、PyTorch 2.14.1+cpu，版本明细在 requirements-lock.txt。首次在新机器准备环境：

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe -r experiments/w01-attention/requirements-lock.txt
```

从学习仓库根目录运行：

```powershell
.venv\Scripts\python.exe experiments/w01-attention/run.py
.venv\Scripts\python.exe lessons/week01/build.py
node lessons/week01/check_interactive.cjs
```

node 命令仅用于交互逻辑检查；打开交互 HTML 本身无需安装 Node.js。图表默认使用 Windows 微软雅黑；其他平台需提供 Noto Sans CJK SC 字体，才能正常显示中文。build.py 依赖本仓库内的数据和资源，不需要访问旧笔记或外部模型源码。

## 文件与证据

| 文件 | 内容 |
| --- | --- |
| attention.py | 从头编写的可读小型 decoder，核心缓存逻辑 |
| run.py | 固定种子的实验、断言和数据导出 |
| data/checks.json | 25 项检查的误差、阈值、环境、种子、代码 SHA-256 |
| data/attention_example.json | 手算例子的 Q/K/V、分数、权重、输出 |
| data/capacity.csv | 216 组理论 KV 载荷配置；不含模型权重及运行时开销 |
| data/work.csv | prompt=128、生成=128 的理论工作量；含共同 prefill |
| data/prior-softmax-check.json | 旧 Pyre softmax 的复核记录，与 25 项主实验分开 |
| ../../lessons/week01/figures/ | 图表 PNG 与矢量 PDF |

正确性正例使用 torch.testing.assert_close，atol=rtol=1e-10。4 个负例的误差应大于 1e-5，这表示故障被发现。25 项检查不是 25 项性能 benchmark。

数据类型保持明确：Attention 数值与 logits 误差来自 CPU 实际运算；容量与工作量按公式导出。未记录墙钟时间，不提供吞吐或 TPOT 数据。

## 实现边界

- 使用普通全因果注意力，无 padding、无滑动窗口、无 tensor parallel、无分页 KV、无多请求共享。
- 两层、D=32、Hq=4，Hkv 分别为 4、2、1；词表为 41。无 dropout。
- RoPE 采用相邻维配对，不声称与 MiniMind 或 Llama checkpoint 的特征布局直接兼容。
- 不训练、不加载真实权重、不测精度压缩、不测试 GPU kernel 或完整服务引擎。
- 缓存形状是 [B,Hkv,S,d]；所有实例中的序列长度相同。
- 输出 8 个 token 的实验为 1 次 prefill + 7 次 decode，最后 token 未送回，最终缓存长度为 4+8-1=11。

## 复核旧 Pyre softmax（可选）

```powershell
.venv\Scripts\python.exe experiments/w01-attention/check_prior_softmax.py --source D:\workdir\2src\pyre\practice\02.softmax.py
```

该脚本只适用于已经阅读确认的本地练习文件：调用原 my_softmax 以复现返回值类型错误，再检查补上 .values 的公式。它不会修改旧文件。离线教材与主实验无需此旧目录。

## 你需要自己完成的扩展

1. 先解释 dim=2 为什么是缓存序列轴，再尝试不同的 batch/长度。
2. 把一个 chunk 改成多个 chunk，预测每次缓存的形状。
3. 在 wrong_mask、wrong_position 中任选一例，说明首次产生错误的算子。
4. 把因果 mask 关闭，构造“改变未来 token 导致过去输出变化”的反例。
5. 记录新增检查及结果，不将助手的现成结果当作自己的学习验收。
