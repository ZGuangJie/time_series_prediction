# 使用指南

本文档提供详细的使用说明和示例。

## 目录

1. [快速开始](#快速开始)
2. [数据准备](#数据准备)
3. [配置说明](#配置说明)
4. [训练模型](#训练模型)
5. [模型预测](#模型预测)
6. [高级用法](#高级用法)
7. [性能调优](#性能调优)

---

## 快速开始

### 方式一：一键启动（推荐）

```bash
python quick_start.py
```

这个脚本会自动完成：
1. 生成示例数据
2. 训练模型
3. 执行预测
4. 生成可视化结果

### 方式二：分步执行

```bash
# 1. 生成示例数据
python generate_sample_data.py

# 2. 训练模型
python train.py --config config/config.yaml

# 3. 预测
python predict.py \
    --config config/config.yaml \
    --model_path checkpoints/best_model.pth \
    --mode file \
    --input_file data/raw/ship_data.csv \
    --output_file results/predictions.csv \
    --visualize
```

---

## 数据准备

### 数据格式要求

CSV文件必须包含以下列：

| 列名 | 类型 | 说明 | 示例 |
|------|------|------|------|
| longitude | float | 经度 | 121.5000 |
| latitude | float | 纬度 | 31.2000 |
| speed | float | 航速（节） | 15.5 |
| course | float | 航向（度，0-360） | 45.0 |

可选列：
- `timestamp`: 时间戳
- `ship_id`: 船舶ID（多船舶场景）

### 数据示例

```csv
longitude,latitude,speed,course
121.5000,31.2000,15.5,45.0
121.5010,31.2008,15.3,44.5
121.5020,31.2016,15.8,44.0
121.5030,31.2024,16.0,43.5
```

### 数据质量建议

1. **时间间隔**：建议固定间隔（如10分钟）
2. **数据量**：至少1000条记录，推荐5000+
3. **完整性**：尽量减少缺失值
4. **异常值**：已有自动处理机制，但提前清洗更好

---

## 配置说明

配置文件位置：`config/config.yaml`

### 关键配置项

#### 1. 数据配置

```yaml
data:
  raw_data_path: "data/raw/ship_data.csv"  # 原始数据路径
  lookback_window: 6    # 回顾窗口（使用过去6个时间步预测）
  prediction_window: 3  # 预测窗口（预测未来3个时间步）
  train_ratio: 0.7      # 训练集比例
  val_ratio: 0.2        # 验证集比例
  test_ratio: 0.1       # 测试集比例
```

**调优建议**：
- `lookback_window`：3-10，取决于时序依赖长度
- `prediction_window`：1-5，更大的值会降低精度

#### 2. 模型配置

```yaml
model:
  model_type: "LSTM"  # LSTM, GRU, Transformer
  
  # LSTM/GRU配置
  lstm:
    hidden_size: 64     # 隐藏层大小（32-128）
    num_layers: 2       # 层数（1-3）
    dropout: 0.2        # Dropout（0.1-0.5）
    bidirectional: false
  
  # Transformer配置
  transformer:
    d_model: 64         # 模型维度
    nhead: 4            # 注意力头数
    num_layers: 2       # Transformer层数
    dim_feedforward: 256
    dropout: 0.2
```

**模型选择**：
- 小数据（<3000）：LSTM with `hidden_size=32, num_layers=1`
- 中等数据（3000-10000）：LSTM with `hidden_size=64, num_layers=2`（默认）
- 大数据（>10000）：Transformer or LSTM with `hidden_size=128, num_layers=3`

#### 3. 训练配置

```yaml
training:
  batch_size: 64          # 批次大小
  num_epochs: 100         # 训练轮数
  learning_rate: 0.001    # 学习率
  weight_decay: 0.001     # L2正则化
  device: "cuda"          # cuda或cpu
  
  optimizer: "adam"       # adam, adamw, sgd
  
  # 学习率调度器
  scheduler:
    type: "ReduceLROnPlateau"
    patience: 10
    factor: 0.5
  
  # 早停
  early_stopping:
    patience: 20
    min_delta: 0.0001
  
  # 损失函数
  loss_function: "weighted_mse"  # mse, mae, huber, weighted_mse
  time_step_weights: [1.0, 0.8, 0.6]  # 时间步权重
```

**调优建议**：
- CPU训练：`batch_size=32, num_epochs=50`
- GPU训练：`batch_size=64-128, num_epochs=100-200`
- 过拟合：增加`dropout`、`weight_decay`，减少`hidden_size`
- 欠拟合：增加`hidden_size`、`num_layers`，减少`dropout`

---

## 训练模型

### 基本训练

```bash
python train.py --config config/config.yaml
```

### 监控训练

#### 方法1：查看日志

```bash
tail -f logs/training.log
```

#### 方法2：TensorBoard

```bash
tensorboard --logdir=runs/
```

然后访问 http://localhost:6006

### 训练输出

```
time_series_prediction/
├── checkpoints/
│   ├── best_model.pth          # 最佳模型
│   ├── best_history.npy        # 最佳模型训练历史
│   └── epoch_50_model.pth      # 定期保存的检查点
├── logs/
│   └── training.log            # 训练日志
├── runs/
│   └── [timestamp]/            # TensorBoard日志
└── results/
    └── training_history.png    # 训练历史图
```

---

## 模型预测

### 从文件预测

```bash
python predict.py \
    --config config/config.yaml \
    --model_path checkpoints/best_model.pth \
    --mode file \
    --input_file data/raw/ship_data.csv \
    --output_file results/predictions.csv \
    --visualize
```

### 预测结果

#### 控制台输出

```
评估结果:
MAE: 0.003456
RMSE: 0.004789
MAPE: 2.345
R2: 0.967
Distance_Error_km: 0.456
```

#### 输出文件

`results/predictions.csv`:

```csv
sample_id,time_step,true_longitude,true_latitude,pred_longitude,pred_latitude
0,1,121.5100,31.2100,121.5098,31.2102
0,2,121.5110,31.2108,121.5107,31.2110
0,3,121.5120,31.2116,121.5118,31.2115
...
```

#### 可视化结果

- `results/predictions_comparison.png`：预测对比图
- `results/trajectory_comparison.png`：轨迹对比图

---

## 高级用法

### 1. 使用不同模型

#### LSTM（推荐，快速训练）

```yaml
model:
  model_type: "LSTM"
```

#### GRU（更快，性能相近）

```yaml
model:
  model_type: "GRU"
```

#### Transformer（大数据，长序列）

```yaml
model:
  model_type: "Transformer"
```

### 2. 自定义损失权重

强调近期预测：

```yaml
training:
  loss_function: "weighted_mse"
  time_step_weights: [1.0, 0.7, 0.5]  # 第1步权重最高
```

### 3. 鲁棒性测试

```yaml
evaluation:
  robustness_test:
    enabled: true
    noise_level: 0.5  # 航速噪声±0.5节
```

### 4. 批量实验

创建多个配置文件，批量训练：

```bash
for config in config/*.yaml; do
    python train.py --config $config
done
```

---

## 性能调优

### CPU训练优化

```yaml
training:
  batch_size: 32
  num_epochs: 50
  num_workers: 2  # CPU核心数

model:
  lstm:
    hidden_size: 32
    num_layers: 1
```

### GPU训练优化

```yaml
training:
  batch_size: 128
  num_epochs: 200
  num_workers: 4
  device: "cuda"

model:
  lstm:
    hidden_size: 128
    num_layers: 2
```

### 内存优化

如果遇到 OOM (Out of Memory)：

1. 减小 `batch_size`
2. 减小 `hidden_size`
3. 减小 `lookback_window`
4. 使用梯度累积（需修改代码）

### 精度优化

提升预测精度：

1. **数据质量**：
   - 增加数据量
   - 清洗异常值
   - 确保时间间隔一致

2. **模型调优**：
   - 增加模型容量（`hidden_size`, `num_layers`）
   - 调整 `lookback_window`
   - 使用双向LSTM (`bidirectional: true`)

3. **训练策略**：
   - 增加训练轮数
   - 调整学习率
   - 使用学习率调度器
   - 数据增强

---

## 常见问题

### Q: 训练很慢怎么办？

A: 
1. 确认使用GPU：`device: "cuda"`
2. 增加batch_size
3. 减少模型复杂度
4. 使用GRU代替LSTM

### Q: 预测精度不够？

A:
1. 增加数据量
2. 增加训练轮数
3. 调整lookback_window
4. 尝试不同模型

### Q: 如何处理多个船舶？

A:
1. 为每个船舶单独训练模型
2. 或添加船舶ID作为额外特征（需修改代码）

### Q: 如何进行实时预测？

A:
1. 加载训练好的模型
2. 准备最近N个时间步的数据
3. 调用predict.py

---

## 进阶开发

### 添加新特征

1. 编辑 `preprocessing/feature_engineer.py`
2. 修改 `create_derived_features` 方法
3. 更新配置文件中的 `input_features`

### 自定义评估指标

1. 编辑 `utils/metrics.py`
2. 添加新的指标函数
3. 在 `training/evaluator.py` 中调用

### 模型集成

```python
# 加载多个模型
model1 = LSTMModel(...)
model1.load_model('checkpoints/lstm_model.pth')

model2 = TransformerModel(...)
model2.load_model('checkpoints/transformer_model.pth')

# 集成预测
pred1 = model1(X)
pred2 = model2(X)
ensemble_pred = (pred1 + pred2) / 2
```

---

## 技术支持

如有问题，请：
1. 查看 [README.md](README.md)
2. 查看 [CHANGELOG.md](CHANGELOG.md)
3. 提交Issue

---

**祝使用愉快！🚢**

