# Git 维护

本仓库位于 C:\Users\jiang\Desktop\study\inference\inference-study，初始分支为 codex/study-plan。

## 每次学习结束

在 PowerShell 中执行：

```powershell
Set-Location -LiteralPath 'C:\Users\jiang\Desktop\study\inference\inference-study'
git status --short
git diff
# 将日期与文件名替换为本次实际产物，只添加本次相关文件。
git add -- logs/2026-10-07.md PROGRESS.md
git diff --cached
git commit -m "study(w01): record transformer shape exercise"
```

实验提交同时带上运行命令、版本、配置、随机种子、单位和小型结果。大模型、环境目录、大型 trace 与原始性能采样不入库；对外部数据保留来源、校验和与重建/获取步骤。小型脱敏 trace 可按需放在 experiments/ 对应目录。

## 远程仓库

当前没有远程地址，也未推送。以后提供个人远程仓库 URL 再配置 origin，并将当前分支推送到该地址。若远程已有历史，先检查并选择合并方式，不覆盖已有提交。

## 分支与复现

- 普通笔记在当前学习分支提交；独立实验可使用 codex/w11-cache-policy 之类的分支。
- 以一次有意义的学习产物为提交单位，不将模板创建误记为学习完成。
- 实验报告记录执行代码对应的 commit；生成数据后再提交结果，可同时记录运行时是否存在未提交差异。
- 外部源码的提交号与本学习仓库提交号分别记录，不混为一个版本。
