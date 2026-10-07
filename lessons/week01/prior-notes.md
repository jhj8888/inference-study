# 从已有 MiniMind / Pyre 笔记进入第一周

核对日期：2026-10-07。资料只读引用，未修改原目录，也未将整套旧笔记复制到远程仓库。以下映射依据读到的内容，不将“有笔记”当成“已掌握”。

## 对照路线

| 本周内容 | 你的已有资料 | 本次新增的学习重点 |
| --- | --- | --- |
| 模型路径与 block | MiniMind 模型分析笔记 §2、§10–12 | 把每一步的逻辑张量形状补齐，区分 logits 与采样 |
| Attention 手算 | Pyre practice/05.attention.py；05 的 PDF | 添加可见位置、逐行 softmax 及误差核验 |
| 因果掩码 | Pyre practice/09.因果注意力机制详解.pdf | 从无历史的方阵推广到有历史的 [T,P+T] 矩形 |
| GQA | MiniMind 笔记 §6、§7.4.7；Pyre 10 的 PDF | 区分存储中的 Hkv 与运算中的 Hq，不把扩头副本存进缓存 |
| RoPE | MiniMind 的 02.05 可视化笔记 | 检查增量位置从 past 接续，并以故意重置位置的负例验证 |
| KV 拼接 | MiniMind 笔记 §7.4.5–7.4.9、§16 | 跨层因果不变性、首 token 时序、缓存失效与重建 |
| softmax 稳定性 | Pyre practice/02.softmax.py | 修正返回值的使用，并验证大数输入下的稳定结果 |

## 直接打开旧资料

- [MiniMind 完整模型笔记](D:/workdir/2src/minimind/my_notes/02.minimind_model_minimind_analysis.md)
- [RoPE 可视化笔记](D:/workdir/2src/minimind/my_notes/02.05.RoPE旋转位置编码可视化解释.md)
- [MiniMind 当前模型源码](D:/workdir/2src/minimind/model/model_minimind.py)
- [Pyre Attention 练习](D:/workdir/2src/pyre/practice/05.attention.py)
- [Pyre softmax 练习](D:/workdir/2src/pyre/practice/02.softmax.py)
- [Pyre Attention PDF](D:/workdir/2src/pyre/practice/05.scaled_dot_product_attention笔记.pdf)
- [Pyre 因果注意力 PDF](D:/workdir/2src/pyre/practice/09.因果注意力机制详解.pdf)
- [Pyre GQA PDF](<D:/workdir/2src/pyre/practice/10.GroupQueryAttention 技术详解.pdf>)

这些是本机路径，在 GitHub 网页上不能直接读取；换电脑需恢复相应资料。配套教材与最小实验不依赖这些目录即可运行。

## 一个形状差异，不是两个算法

| 操作 | MiniMind 当前源码 | 本周教学模型 |
| --- | --- | --- |
| 紧凑缓存布局 | [B,S,Hkv,d] | [B,Hkv,S,d] |
| 序列拼接轴 | dim=1 | dim=2 |
| KV 扩头时间 | 保存紧凑缓存后、Attention 前 | 保存紧凑缓存后、Attention 前 |
| 历史长度 | cache[0].shape[1] | cache[0].shape[2] |
| 位置 | start_pos 切片 RoPE 表 | past + arange(T) |
| 因果屏蔽 | 当前 chunk 对应的最后 T 列加三角掩码 | j ≤ P+i |

已核对 `model/model_minimind.py`：Attention.forward:111；缓存拼接:120；保存缓存:123；扩头和转置:124；手写掩码:129；start_pos:213；位置表切片:219。源码位于 Git HEAD `4497610ec0a85d2d0a3db488fd4c1e2f12a416ab` 的工作树，工作树文件指纹见 [prior-sources.json](prior-sources.json)；HEAD 本身不保证未提交文件属于该提交。

RoPE 也有布局差异：MiniMind 的 rotate_half 将前后两半配对，本周教学函数采用相邻偶奇维配对。它们需要配套的频率表和特征排列，不能把其中一个函数直接替换进另一个模型后仍期待输出不变。本周只检查每个固定模型自身的缓存等价性。

## Pyre softmax 中需要修正的一处返回值

原文件写为 `x_max = x.max(dim=dim, keepdim=True)`。指定 dim 时返回值包含 values 和 indices，不能直接与 Tensor 相减。应取：

```python
x_max = x.max(dim=dim, keepdim=True).values
# 或 x_max = torch.amax(x, dim=dim, keepdim=True)
e_x = torch.exp(x - x_max)
return e_x / e_x.sum(dim=dim, keepdim=True)
```

已在本周 PyTorch 2.14.1+cpu 环境中复现原函数的 TypeError，并在 float64 输入 `[1000,1001,1002]` 上核验修正后的结果与 torch.softmax 一致：约 `[0.090031,0.244728,0.665241]`。原文件未改动。证据见 [prior-softmax-check.json](../../experiments/w01-attention/data/prior-softmax-check.json)；复核命令见实验说明。

Pyre 的 Attention 练习当前是无 mask 的 scaled dot-product attention，是合理的基础练习；要迁移到本周的因果 decoder，需要增加掩码、头结构、历史 KV 和位置约定。PDF 中 `triu(..., diagonal=1)` 的 True 表示禁止，与本周 `allowed` 的 True 表示允许恰好相反。

## 利用旧基础的 15 分钟自测

不翻笔记，解释：①为什么 MiniMind 的 concat 是 dim=1？②P=5、T=2 时两个 query 能看哪些 key？③刚采样出来的 token 已经在 KVCache 中吗？④仅修改采样温度与修改权重，对旧 KV 的影响有什么区别？

能独立回答就快速复习第 1、2 节，重点练第 3、5、6 节；卡在哪一题，就回到对应旧笔记和新教材核对。保留同样的周验收标准，不因目录里有资料而跳过验收。
