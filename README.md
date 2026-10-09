# Inference Study

12 周、192 小时：从 Attention 和 KVCache 到推理引擎、存储与后训练衔接，完成一个可复现的小型优化项目。

起始日期 **2026-10-07（周三）**；基线结束日期 **2026-12-29（周二）**。当前阶段：**已建档，尚未开始学习**。

远程仓库：[jhj8888/inference-study](https://github.com/jhj8888/inference-study)（私有），学习分支为 codex/study-plan。

## 从这里开始

- [第一周图文教材](lessons/week01/README.md)与[离线交互版](lessons/week01/index.html)：7 张图、3 个交互演示、CPU 正确性实验、练习和答案。
- [第二周推理性能分析](lessons/week02/README.md)与[离线交互版](lessons/week02/index.html)：7 张图、4 个交互实验、估算脚本与[实验协议](experiments/w02-performance/protocol.md)。
- [第三周 vLLM 请求生命周期教材](lessons/week03/README.md)与[离线交互版](lessons/week03/index.html)：6 张图解、3 个交互演示、完整本地源码快照、45 个源码锚点与三请求模拟；附[资料阅读导航](sources/w03-reading-route.md)。
- [论文阅读索引](papers/README.md)：DeepSeek-V4.1-Flash 与 FlashAttention 1–4，提供中文译本、已收录原文、配套图表及阅读顺序。
- [衔接已有 MiniMind / Pyre 笔记](lessons/week01/prior-notes.md)：按已有内容选择复习入口。
- [今天的学习入口](logs/2026-10-07.md)：Transformer 推理路径与 Q/K/V 张量形状，2 小时。
- [当前进度](PROGRESS.md)：实际完成情况、卡点、下一次任务。
- [完整 12 周计划](plan/README.md)与[逐日日历](plan/schedule.csv)。
- [本地资料与源码导航](sources/README.md)：导读、mini-sglang、SGLang、vLLM、LMCache、Mooncake。
- [Git 使用方式](GIT_WORKFLOW.md)。

## 每次学习

1. 先看 PROGRESS.md，再说“开始今天的学习”。当天任务按已通过的验收推进。
2. 按 15/35/50/20 分钟完成回忆、原理、实践与记录；周末各 3 小时。
3. 使用 [日志模板](templates/daily.md)记录“做了什么、产物、卡在哪里”。
4. 周日使用 [周验收模板](templates/weekly-review.md)；实验使用 [实验模板](templates/experiment.md)。
5. 更新进度并提交相关笔记、代码和小型结果数据。

## 论文资料

`papers/` 保存与 Attention、KVCache 和推理性能相关的论文资料，可配合每周教材阅读。

| 专题 | 已收录内容 | 学习关注点 |
| --- | --- | --- |
| [DeepSeek-V4.1-Flash](papers/DeepSeek-V4.1-Flash/阅读说明.md) | 带批注的原文 PDF、中文 Markdown / PDF、Confluence 版 DOCX、配套图片 | 译本中的 KV 压缩、跨层复用与持久化缓存管理，衔接第 1、6–8 周 |
| [FlashAttention 1–4](papers/README.md#flashattention-系列) | 四代中文 Markdown / PDF 与配套图片；第 1–3 代另有原文 PDF 和阅读说明 | 从 IO 感知、并行划分到异步流水线与硬件适配，衔接第 1–2 周并作为后续进阶阅读 |

各篇入口与建议顺序见[论文阅读索引](papers/README.md)。Markdown 依赖同目录下的 `assets/` 图片；论文实验数据按原文条件理解，本项目实测结果另见 `experiments/` 和 `reports/`。

## 目录

| 目录 | 内容 |
| --- | --- |
| plan/ | 原始学习模块与逐日预算 |
| lessons/ | 图文教材、离线交互页面及可重建图表 |
| [papers/](papers/README.md) | 论文原文、中文译本、阅读说明与配套图表 |
| logs/ | 每次学习的真实记录 |
| notes/ | 原理推导、源码调用链和对照笔记 |
| experiments/ | 最小实验与复现命令 |
| reports/ | 周产物与最终技术报告 |
| templates/ | 日志、周验收、实验和源码阅读模板 |
| sources/ | 源码位置、入口及所选文件的 SHA-256 基线 |
| scripts/ | 源码指纹记录与核验工具 |

源码仍放在原目录，本仓库保存学习产物和引用信息。vLLM 已额外保存完整本地归档与锁文件；归档被 Git 忽略，需单独备份。其他源码目前只留选定入口指纹，上游提交号尚未确认。详见源码导航中的覆盖范围说明。

预算合计：启动周 12 小时 + 11 个完整周 × 16 小时 + 最后两个工作日 × 2 小时 = **192 小时**。未通过验收则补基础并调整后续日期。
