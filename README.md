# Inference Study

12 周、192 小时：从 Attention 和 KVCache 到推理引擎、存储与后训练衔接，完成一个可复现的小型优化项目。

起始日期 **2026-10-07（周三）**；基线结束日期 **2026-12-29（周二）**。当前阶段：**已建档，尚未开始学习**。

远程仓库：[jhj8888/inference-study](https://github.com/jhj8888/inference-study)（私有），学习分支为 codex/study-plan。

## 从这里开始

- [第一周图文教材](lessons/week01/README.md)与[离线交互版](lessons/week01/index.html)：7 张图、3 个交互演示、CPU 正确性实验、练习和答案。
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

## 目录

| 目录 | 内容 |
| --- | --- |
| plan/ | 原始学习模块与逐日预算 |
| lessons/ | 图文教材、离线交互页面及可重建图表 |
| logs/ | 每次学习的真实记录 |
| notes/ | 原理推导、源码调用链和对照笔记 |
| experiments/ | 最小实验与复现命令 |
| reports/ | 周产物与最终技术报告 |
| templates/ | 日志、周验收、实验和源码阅读模板 |
| sources/ | 源码位置、入口及所选文件的 SHA-256 基线 |
| scripts/ | 源码指纹记录与核验工具 |

源码仍放在原目录，本仓库保存学习产物和引用信息。所选入口文件已留存指纹；上游提交号尚未确认。详见源码导航中的覆盖范围说明。

预算合计：启动周 12 小时 + 11 个完整周 × 16 小时 + 最后两个工作日 × 2 小时 = **192 小时**。未通过验收则补基础并调整后续日期。
