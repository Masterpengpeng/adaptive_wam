# 实验记录导航

本目录用于追踪 Adaptive Inference for World Action Models 的实验进度与效果。
根目录的 [EXPERIMENT_LOG.md](../../EXPERIMENT_LOG.md) 仍是长期实验协议和背景说明的来源；本目录是更便于日常更新的状态面板与结果记录。

## 文档

- [当前阶段](./current_stage.md)：唯一的当前阶段状态页，记录本周/当前阶段、完成度、阻塞项和下一步验收条件。
- [实验进度日志](./experiment_progress.md)：按日期追加可验证的实验执行记录。
- [实验效果与结果](./experiment_results.md)：汇总已测量数据、解读边界和报告口径。
- [项目实验协议](../../EXPERIMENT_LOG.md)：研究假设、变量定义、实验规则和历史上下文。

## 更新规则

1. 开始或结束阶段时，先更新 `current_stage.md`。
2. 每次运行实验后，在 `experiment_progress.md` 追加一条记录：目标、命令、环境、结果路径、状态和异常。
3. 只有从原始输出中核实的数据才写入 `experiment_results.md`；负面结果与失败运行也应记录。
4. 不覆盖旧结果。每次正式重复实验使用新的、带日期或 run ID 的输出目录。
5. 区分 smoke test、pilot 与正式评测。一次配置/一次种子的结果不能表述为统计显著或泛化结论。
6. checkpoint、环境状态等大文件保留在本地数据目录；Markdown 中记录路径、版本和校验值，不复制权重。

## 实验记录模板

```text
日期 / Run ID:
阶段:
目标:
代码版本 / Git commit:
环境与硬件:
Checkpoint 与校验值:
命令:
参数（任务、种子、配置、预算）:
结果目录:
结果摘要:
异常与偏差:
结论:
下一步:
```
