# 特征自动化筛选框架

**目的**：能够根据模型类型（机器学习 or 评分卡模型）及各变量模块自动化清洗数据、个性化设置缺失值阈值、同时逐步拆分，从整体 -> TRN v.s OOT -> by 月，最终输出效果好且稳定的变量。

## 零、代码运行
修改完config文件后，直接运行`feature_select.py`文件即可

## 一、config文件设置

### 数据path：
- 如果Y标与特征不在同一个表中，则需填写`label_path`和`feature_path`，默认以`serial_id`拼接两张表
- 如果在同一表中，仅需填写`merge_file_path`，`label_path`和`feature_path`需为None

### 参数设置：
- `target_name`：Y值名称
- `train_month`：设置训练集的月份list
- `oot_month`: 设置OOT集的月份list
- `monthly_base_month`：用于设置by 月PSI的base月份
- `y_label_list`：存储各类Y标的list
- `to_drop_list`：非特征列
- `time_column`：数据中的事件列
- `result_path`：存储结果的路径
- `missing_rate`：默认变量缺失率阈值(不同维度的阈值可以指定)
- `iv_rate`：变量IV阈值
- `psi_rate`：变量PSI阈值
- `Importrance_rate`：变量重要性阈值
- `target_corr`：与目标变量相关性阈值
- `var_corr`：变量间相关性阈值
- `if_scorecard`：是否为评分卡模型（填写true or false）
- `distribution_std`：Train v.s OOT的标准差阈值
- `iv_month_std`：by 月看变量IV的标准差阈值
- `PSI_month_std`：by 月看变量PSI的标准差阈值
- `mean_month_std`：by 月看变量均值的标准差阈值
- `dimensions`：特征维度及变量名（各维度的缺失率阈值在这里指定）

## 二、整体框架设计

### 2.1 初筛：分模块观察整体数据情况

**输入**：整体数据  
**输出**：满足各阈值的变量及对应的观察指标  
**观察指标**：IV、PSI、特征重要性、target_corr(按照绝对值筛！)  
**注**：需要对每个观察指标保留阈值参数；相关性强的变量，保留其中高IV值变量。  
**参数设置**：missing_rate、iv_rate、PSI_rate、Importrance_rate、target_corr_rate、var_rate  

**具体步骤**：
1. 数据预处理：对类别变量进行标签编码（注：适用于树模型，不适用于逻辑回归）
2. 剔除不满足缺失率要求的变量
3. 计算各变量的IV\变量重要性\与目标变量的相关性\PSI（使用数据中最新一个月的数据座位OOT），同时剔除不满足要求的变量
4. 对于变量间相关性较高的情况，仅保留IV值较高的变量，避免共线性问题
5. 保存初筛结果

### 次筛：TRN v.s OOT

**输入**：切分TRN和OOT后的数据  
**输出**：按Train数据进行等频分箱（10等分）和OOT数据列出各变量分箱后的bad count\bad rate\Lift\各分箱占比（perc）  
**观察指标**：bad_rate、LIFT  
**筛选条件**：TRN及OOT分箱后bad_rate和Lift指标均需满足单调性、TRN及OOT分布差异不能过大(目前是针对Lift的单调性进行筛选)  
**注**：对于评分卡模型，需要额外看WOE  
**参数设置**：if_scorecard\tolerance(单调性容忍度，还没想好是咋设置，暂时设置为None)\distribution_std  

### 终筛：by 月观察每个变量的稳定性

**输入**：整体数据  
**输出**：满足稳定性各变量的by 月的IV值、PSI、及变量均值  
**观察指标**：IV值、PSI、变量均值  
**筛选条件**：by月满足方差阈值  
**参数设置**：iv_month_std\PSI_month_std\mean_month_std
