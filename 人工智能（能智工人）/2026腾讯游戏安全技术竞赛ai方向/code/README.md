## 项目说明

本项目用于完成游戏行为决策预测任务，目标是根据游戏日志中主玩家及周围玩家的状态信息，预测主玩家的具体动作行为，并进一步映射得到意图决策结果。

### 目录说明

- `data_analysis.py`  
  
  对结构化后的训练数据进行分析，包括缺失值、字段类型、类别分布等。
  
- `data_extract.py`  
  
  从原始训练日志中提取静态快照，生成结构化训练数据。
  
- `data_process.py`  
  
  对提取后的数据进行基础清洗和初步特征处理。
  
- `data_process_geo.py`  
  
  在基础特征上加入几何关系特征和其他增强特征。
  
- `preliminary_training.py`  
  
  用于验证基础特征处理后的模型效果。
  
- `preliminary_training_geo.py`  
  
  用于验证加入几何增强特征后的模型效果。
  
- `weight_test.py`  
  
  测试不同类别权重方案对模型结果的影响。
  
- `weight_search.py`  
  
  在手动权重附近做小范围搜索，寻找最优权重组合。
  
- `final.py`  
  
  最终训练与预测脚本，用于生成模型和原始预测结果。
  
- `format.py`  
  
  将预测结果整理成符合比赛官方要求的提交格式。
  
- `final_outputs`  

  保存最终模型、预测结果和提交文件。

### 整体流程

1. 使用 `data_extract.py` 从原始日志中提取静态快照  
2. 使用 `data_analysis.py` 对数据进行分析  
3. 使用 `data_process.py` 完成基础特征处理  
4. 使用 `data_process_geo.py` 加入几何增强特征  
5. 使用 `preliminary_training.py` 和 `preliminary_training_geo.py` 验证特征效果  
6. 使用 `weight_test.py` 和 `weight_search.py` 确定最终类别权重  
7. 使用 `final.py` 训练最终模型并生成预测结果  
8. 使用 `format.py` 输出符合要求的提交文件  

### 模型方案

项目最终采用 CatBoost 进行多分类训练，先预测具体动作行为，再映射得到意图决策。

动作行为包括：

- 开火
- 放技能
- 丢雷
- 搜
- 救援

意图决策包括：

- 交战
- 避战

映射关系为：

- 搜、救援 -> 避战
- 开火、放技能、丢雷 -> 交战

### 运行环境

建议使用 Python 3.10 及以上版本。

主要依赖：

```bash
catboost
pandas
numpy
openpyxl