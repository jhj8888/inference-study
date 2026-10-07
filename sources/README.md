# 本地资料与源码导航

本仓库的父目录为 C:\Users\jiang\Desktop\study\inference。以下路径相对此父目录。catalog.json 记录相对此学习仓库的源码位置和所选入口文件。

## 阅读顺序

| 学习阶段 | 资料或源码 | 本地目录与入口 |
| --- | --- | --- |
| W01–W02 | zero-to-sglang 导读 | zero-to-sglang-main/course-material/ch/part1/：第 1、2、4、5 章 |
| W01、W04–W05 | mini-sglang | zero-to-sglang-main/mini-sglang-main/python/minisgl/：models/llama.py、layers/attention.py、scheduler/scheduler.py、kvcache/radix_cache.py |
| W03–W04 | vLLM | vllm-main/vllm/v1/：engine/core.py、core/sched/scheduler.py、core/kv_cache_manager.py |
| W03 主线，W04 继续 | 中文 vLLM 学习手册 | vllm-learning-book-main/：01-overview/、03-code-walkthrough/、07-hands-on/ |
| W03 补充，W02 可回查 | 英文 vLLM 幻灯教程 | tutorials-vllm-main/：06-openai-api-standard/、12-input-output-tokenization/、14-continuous-batching/ 等目录下 slides.md |
| W05 | SGLang | zero-to-sglang-main/sglang-main/python/sglang/srt/：managers/scheduler.py、mem_cache/radix_cache.py |
| W06 | LMCache | LMCache-dev/lmcache/v1/cache_engine.py、v1/storage_backend/storage_manager.py、integration/vllm/lmcache_connector_v1.py |
| W07 | Mooncake | Mooncake-main/README.md、mooncake-transfer-engine/src/transfer_engine.cpp；Store 的具体取回链在该阶段进一步定位 |

W08 沿用缓存、I/O 和传输的证据。W09–W10 的后训练框架尚未选定；到该阶段再核实实际版本和同步接口。不要假定以上仓库已包含所需的完整训练闭环。

第三周的逐日章节、可点击的本地文件入口与版本差异见 [W03 阅读导航](w03-reading-route.md)。这两份资料用于备课和阅读辅助；真实调用链仍须核实源码。

## 已核实的定位锚点

行号为 2026-10-07 建档时的本地文件定位；更新源码后须重查。这些是阅读入口，还不是已经验证完毕的端到端调用链。

| 项目 | 文件 | 符号与当时行号 |
| --- | --- | --- |
| mini-sglang | python/minisgl/models/llama.py | LlamaDecoderLayer、LlamaModel、LlamaForCausalLM（按符号查找） |
| vLLM | vllm/v1/engine/core.py | EngineCore:111；add_request:489；step:633 |
| vLLM | vllm/v1/core/sched/scheduler.py | Scheduler:79；schedule:553 |
| vLLM | vllm/v1/core/kv_cache_manager.py | KVCacheManager:131；allocate_slots:371 |
| SGLang | python/sglang/srt/managers/scheduler.py | Scheduler:436 |
| SGLang | python/sglang/srt/mem_cache/radix_cache.py | RadixCache:280；match_prefix:354；insert:414 |
| LMCache | lmcache/v1/cache_engine.py | LMCacheEngine:83；store:388；retrieve:801；lookup:1157 |
| Mooncake | mooncake-transfer-engine/src/transfer_engine.cpp | registerLocalMemory:183/804；submitTransfer:208/904，需区分构建分支 |

LMCache 的设计文档规则要求先查 docs/design/ 中对应模块路径；先从 docs/design/README.md 了解映射。集成、后端和多进程实现会随版本变化，不将入口列表当成完整实现说明。

导读 README_zh.md 中部分进阶章节仍标为筹备中；其目录或完成标记不能替代实际内容与源码核实。第一章主要介绍原始 Transformer，学习 decoder-only 推理时须明确架构差异。

## CodeGraph

建档时 Mooncake-main 和 zero-to-sglang-main/sglang-main 有 .codegraph/。对这两个目录理解或定位代码时，先调用 codegraph_explore 并传入对应 projectPath，或者在对应源码根目录运行 codegraph explore。此次已用 CodeGraph 定位相关入口。

其他已列源码根目录当时无 .codegraph/，直接使用 rg 与文件读取即可，不自动创建索引。每次工作前重新检查目录是否存在。

## 版本与指纹的边界

已检查的 8 个资料/源码根目录（zero-to-sglang 导读、mini-sglang、SGLang、vLLM、两份 vLLM 教程、LMCache、Mooncake）均无顶层 .git，因此各目录实际上游提交号记为 null（未知），不从目录名 main/dev 推断版本。中文 vLLM 手册在 source.lock.json 声明的被讲解源码版本单独记在 W03 阅读导航中，它不等于本地 vllm-main 的已验证提交号。

当前入口基线为 [baseline-2026-10-07-w03-guides.json](baseline-2026-10-07-w03-guides.json)，覆盖新增教程和扩充后的 vLLM 阅读入口；原 baseline-2026-10-07.json 保留不变，代表最初 6 项资料的选定文件。入口基线只记录选定文件的 SHA-256 与字节数，不能证明其他文件没变，也不能凭哈希恢复源码。

**vLLM 已完成完整本地内容锁定**：归档 `artifacts/raw/vllm-local-2026-10-07.zip`，对应[逐文件锁记录](vllm-local-2026-10-07.lock.json)，覆盖 7,289 个文件和全部目录。归档与当前原目录已逐项核验；上游 commit 和最初下载时间仍未知。归档被 Git 忽略，换电脑须单独复制。第三周教材的静态源码结论对应这份本地快照；其他项目完整版本锁定仍为待办。

进入 W03、正式调用链分析或性能实验前：

1. 优先取得与阅读目标相符的上游仓库并固定真实 commit；保留现有目录，以便核对导读所用版本。
2. 如果继续使用解压源码，保存可恢复的完整源码归档及 SHA-256，注明来源与获取时间；对实验涉及的本地改动留存补丁。
3. 补记模型、权重、tokenizer、依赖、配置及构建选项；学习仓库 commit 与上游 commit 分开记。
4. 新版本单独建立基线，不覆盖历史指纹。已有实验仍指向其原版本。

## 核验命令

在学习仓库根目录执行（只用 Python 标准库）：

    py -3.11 scripts/source_snapshot.py verify

第三周完整 vLLM 快照核验另执行：

    py -3.11 scripts/full_source_snapshot.py verify

它同时核对归档 SHA、归档中的全部文件以及原目录清单/字节；并非只检查入口文件。

默认对照当前 W03 扩充基线。旧基线与当前 catalog 的项目及文件集合不同，直接用它核验当前 catalog 会报告差异；回查历史时应使用同一次 Git 提交中的 catalog、脚本及基线。

源码确实需要更换时，先检查差异，再保存新名称的基线，例如：

    py -3.11 scripts/source_snapshot.py capture --observed-on 2026-10-08 --output sources/baseline-2026-10-08.json

这里的日期仅为命令示例，实际使用时替换为观察日期。capture 使用独占创建，拒绝覆盖已有文件。

若换电脑，将 sources/local-paths.example.json 复制为 sources/local-paths.json，修改对应绝对路径；该个人配置已被 Git 忽略。或者按 catalog.json 的同级目录结构放置源码。

选定文件哈希使用原始字节，换行符改变也会被报告。源码哈希核验不等于通过学习验收，也不等于通过引擎运行测试。
