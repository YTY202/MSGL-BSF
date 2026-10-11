# MSGL-BSF 主体代码

本目录包含 MSGL-BSF 的模型结构、训练入口、数据处理和评估脚本。

## 推荐运行方式

进入整理后的 `主体代码` 目录并运行一键入口：

```powershell
cd "C:\Users\15493\Desktop\第二篇-DFFRGM-gaijin\主体代码"
python code\run_msgl_bsf.py
```

默认使用 Dataset 1，并自动加载固定的 MSGL-BSF 配置。

运行唾液代谢物数据集：

```powershell
python code\run_msgl_bsf.py --data_dir data2_saliva_mesh
```

也可以在上一级项目根目录执行 `python 主体代码\code\run_msgl_bsf.py`。入口脚本会根据自身位置定位数据与结果目录，不依赖当前工作目录。

## 文件说明

- `run_msgl_bsf.py`: 一键运行入口，封装当前最终模型配置。
- `train_liquid_neighbor_sequence.py`: MSGL-BSF 主训练脚本，执行多尺度邻域序列构造、双向双速率液态编码和关联预测。
- `model.py`: 模型主体及结构表示、语义表示和特征融合模块。
- `layers.py`: 液态单元、序列编码和其他神经网络层。
- `liquid_neighbor_sequence.py`: 相似度排序邻域和伪时间序列处理组件。
- `data_loader.py`: 数据集读取和预处理工具。
- `param.py`: 模型参数定义和命令行参数设置。
- `utils.py`: 数据划分、相似度特征构造、指标计算及绘图等公共函数；主体代码不再依赖已删除的矩阵补全训练脚本。
- `evaluate_cross_dataset.py`: 使用已保存模型进行跨数据集评估。
- `predict_cross_dataset_liquid.py`: 跨数据集预测和结果保存。

## BioMedBERT

BioMedBERT 语义编码脚本位于上一级目录的 `bert/`：

- `../bert/build_bert_semantic_corpus.py`: 构建代谢物和疾病文本语料。
- `../bert/LLMEncoder.py`: 加载 BioMedBERT 并生成语义特征。

训练时通过 `--bert_model_path` 指定 BioMedBERT 模型目录。服务器示例路径为：

```text
/root/BiomedBERT
```

## 数据和结果

整理后的数据副本位于：

```text
../data/data1
../data/data2
../data/data2_saliva_mesh
```

训练结果默认保存到项目根目录的 `result/`，训练日志保存到 `training_Log/`。

## 重要说明

本目录是整理后的主体代码副本。原始项目中的 `code/` 目录仍然保留，用于兼容已有的 PyCharm 远程配置和历史运行命令。
