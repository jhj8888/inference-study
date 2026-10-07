# 第一周教材交付检查

日期：2026-10-07。此记录属于助手备课验证，不是学习者的验收记录，不计入学习时长。

## 交付内容

- [完整图文讲义](../lessons/week01/README.md)：启动周安排、张量形状、Attention 手算、增量掩码、KVCache、头结构、容量、实验与闭卷验收。
- [离线交互教材](../lessons/week01/index.html)：7 张内嵌图、3 个交互演示，正文阅读不依赖联网或 Python。
- [最小 CPU 实验](../experiments/w01-attention/README.md)：两层 decoder、MHA/GQA/MQA、单 token 和 chunk 输入、RoPE、缓存失效负例。
- [旧笔记衔接](../lessons/week01/prior-notes.md)：MiniMind / Pyre 对照阅读、缓存布局与 RoPE 约定，以及旧 softmax 练习的复核。

## 已完成的验证

| 验证 | 结果 | 证据 |
| --- | --- | --- |
| CPU 数值正确性 | 21 项正例通过，最大绝对误差 1.110e-15；4 项故意故障均检出 | [checks.json](../experiments/w01-attention/data/checks.json) |
| JS/Python 对照 | 三个 query 的权重与输出一致；容量公式、掩码切换、decode 状态与预算边界通过 | `node lessons/week01/check_interactive.cjs` |
| 浏览器 Attention | Q0 无掩码输出 [1.6044,0.7967]；开启掩码恢复 [1.0000,0.0000] | 实际浏览器控件操作 |
| 浏览器 decode | prefill 后缓存 4 个位置；生成 4 个 token 后缓存 7 个位置，“下一步”禁用 | 实际浏览器控件操作 |
| 浏览器容量 | B=8、S=8192、MHA Hkv=32 时为 32 GiB；16 GiB 净预算下提示超出，最多 4 个同长请求 | 实际浏览器控件操作 |
| 练习答案 | 折叠/展开正常；行内公式与数值正常渲染 | 实际浏览器控件操作 |
| 浏览器错误 | 检查时无 console error/warn | 浏览器日志 |
| 图表与排版 | 构建生成 7 组 PNG/PDF；检查容量图、路径图及浏览器窄面板排版 | [浏览器截图](screenshots/week01-browser.jpg) |
| 离线资源 | 7 张图片全部内嵌；无外部脚本/样式依赖；HTML 约 1.8 MB | 对生成 HTML 的静态检查 |
| 链接与数据 | 文档本地链接与目录锚点存在、ID 唯一、容量 CSV 216 行 | 文档静态检查 |
| 源码证据 | 实验代码与结果内记录的 SHA-256 一致；3 个 mini-sglang 文件及 8 个旧资料文件指纹一致 | 两份教材 sources JSON 及 checks.json |
| 旧 softmax | 原函数复现 TypeError；补上 `.values` 的公式与 torch.softmax 一致 | [独立复核记录](../experiments/w01-attention/data/prior-softmax-check.json) |

## 适用边界

容量、计算工作量为理论计算；Attention 数值与 logits 误差为 CPU float64 实测。没有 GPU 延迟、吞吐或加速比实测。随机初始化的小模型用于验证计算正确性，不评价生成质量。

mini-sglang 来自本地源码快照，只记录所读文件的指纹，未声称固定完整上游版本。旧笔记只核对列明的内容范围，未修改原目录。离线教材的核心正文与交互可单文件使用；源码、数据、日志链接需保留仓库目录，旧资料链接依赖原本机路径。
