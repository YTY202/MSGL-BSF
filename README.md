# 主体代码

这里保存 MSGL-BSF 的主体实现，按功能分为 `code`、`data` 和 `bert` 三个子目录。

- `code/`: 模型、训练、评估和通用工具
- `data/`: 数据集位置说明；实际数据仍保留在项目根目录
- `bert/`: BioMedBERT 语义编码相关脚本

主体训练入口位于 `code/train_liquid_neighbor_sequence.py`，一键入口位于
`code/run_msgl_bsf.py`。BioMedBERT 相关脚本位于 `bert/`，数据位置说明位于 `data/`。
