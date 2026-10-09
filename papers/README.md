# 论文阅读索引

[返回项目首页](../README.md)

本目录收录 DeepSeek-V4.1-Flash 和 FlashAttention 1–4 的学习资料。下表按仓库已有文件和译本内容整理，便于查找原文、中文译本与配套说明。

## DeepSeek-V4.1-Flash

主题：译本中的因果编码器—解码器（CED）、压缩稀疏注意力 2（CSA2）、跨层 KV 复用、FP4 KV 缓存，以及持久化缓存与 SWA 有界重放。

| 资料 | 入口 |
| --- | --- |
| 阅读说明 | [译本格式与阅读方式](DeepSeek-V4.1-Flash/阅读说明.md) |
| 中文 Markdown | [DeepSeek-V4.1-Flash：突破 KV 缓存压缩的极限](DeepSeek-V4.1-Flash/DeepSeek-V4.1-Flash_中文译本.md) |
| 中文 PDF | [下载 / 阅读](DeepSeek-V4.1-Flash/DeepSeek-V4.1-Flash_中文译本.pdf) |
| 带批注的原文 PDF | [Pushing the Limits of KV Cache Compression](DeepSeek-V4.1-Flash/DeepSeek-V4.1-Flash%20Pushing%20the%20Limits%20of%20KV%20Cache%20Compression-with-annotations.pdf) |
| Confluence 版 DOCX | [下载文档](DeepSeek-V4.1-Flash/DeepSeek-V4.1-Flash_中文译本_Confluence版.docx) |
| 配套图表 | [assets/](DeepSeek-V4.1-Flash/assets/) |

完成[第一周 KVCache 基础](../lessons/week01/README.md)后，可先读译本第 2 节“架构”，辨认不同缓存的用途与容量口径。学到第 6–8 周的多层缓存、传输和存储时，再读第 3.2 节“推理系统”，整理缓存恢复、数据搬运和重计算之间的取舍。

## FlashAttention 系列

| 资料 | 译本中的主要主题 | 中文译本 | 原文与说明 |
| --- | --- | --- | --- |
| FlashAttention | IO 感知、分块、在线 softmax 与重计算 | [Markdown](FlashAttention/01_FlashAttention/FlashAttention_中文译本.md) · [PDF](FlashAttention/01_FlashAttention/FlashAttention_中文译本.pdf) | [原文 PDF](FlashAttention/01_FlashAttention/FlashAttention.pdf) · [阅读说明](FlashAttention/01_FlashAttention/阅读说明.md) |
| FlashAttention-2 | 减少非矩阵乘法操作、线程块与线程束的工作划分 | [Markdown](FlashAttention/02_FlashAttention-2/FlashAttention2_中文译本.md) · [PDF](FlashAttention/02_FlashAttention-2/FlashAttention2_中文译本.pdf) | [原文 PDF](FlashAttention/02_FlashAttention-2/FlashAttention2.pdf) · [阅读说明](FlashAttention/02_FlashAttention-2/阅读说明.md) |
| FlashAttention-3 | Hopper 异步执行、GEMM 与 softmax 重叠、低精度注意力 | [Markdown](FlashAttention/03_FlashAttention-3/FlashAttention3_中文译本.md) · [PDF](FlashAttention/03_FlashAttention-3/FlashAttention3_中文译本.pdf) | [原文 PDF](FlashAttention/03_FlashAttention-3/FlashAttention3.pdf) · [阅读说明](FlashAttention/03_FlashAttention-3/阅读说明.md) |
| FlashAttention-4 | Blackwell 硬件适配、非矩阵乘法瓶颈、张量内存与流水线设计 | [Markdown](FlashAttention/04_FlashAttention-4/FlashAttention4_中文译本.md) · [PDF](FlashAttention/04_FlashAttention-4/FlashAttention4_中文译本.pdf) | 当前目录收录中文译本与配套图片，原文版本信息见译本开头 |

每篇译本的图片位于各自目录的 `assets/` 下。

## 建议阅读顺序

1. **先完成第一周基础。** 能写出 Attention 的张量形状、因果掩码和 KV 容量公式后，阅读 FlashAttention 的背景与算法部分，画出 Q/K/V 和中间结果在存储层级间的流向。
2. **配合第二周性能分析。** 阅读 FlashAttention 与 FlashAttention-2，结合[第二周教材](../lessons/week02/README.md)中的算术强度和 Roofline，分别记录计算量、数据搬运量和并行划分对性能的影响。
3. **再读硬件相关进阶内容。** 按 FlashAttention-3 → FlashAttention-4 的顺序，画出数据加载、矩阵乘法和 softmax 的依赖及重叠关系；比较论文数据时记录硬件、精度、序列长度、头维度和基线。
4. **结合多层 KV 缓存专题。** 回到 DeepSeek-V4.1-Flash，列清各类缓存驻留的位置、每 token 容量、恢复过程及额外计算，再与[第 6–8 周计划](../plan/README.md)中的成本模型对照。

## 阅读与记录

- 在 GitHub 上可直接阅读 Markdown；保留原文分页和版式的内容可打开 PDF 对照。移动 Markdown 时应同时保留对应的 `assets/` 目录。
- 已有“阅读说明”记录了译本处理方式和部分原文疑点，阅读公式及复现算法前先查看这些说明。
- 论文中的实验数字属于其报告的实验条件。自己的理论估算、CPU 验证和 GPU 实测分别记录，使用[实验模板](../templates/experiment.md)保存环境、配置与证据。
- 阅读后使用[每日记录模板](../templates/daily.md)写下问题、结论、出处和未解决事项；资料收录与教材准备不计为个人学习完成。
