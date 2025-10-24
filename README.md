# 时序数据预测之船舶位置预测 

## 📋 项目简介

这是一个基于PyTorch的完整、可扩展的船舶位置预测系统，采用面向对象设计，支持多种深度学习模型（LSTM、GRU、Transformer）进行时序预测。该项目针对"用经度、纬度、航速、航向预测船舶位置"的需求，构建了多变量时序回归模型。

### 核心特性

- ✅ **完整的数据处理流程**：数据加载、清洗、异常值处理、缺失值填充、标准化
- ✅ **多种模型支持**：LSTM、GRU、Transformer（易于扩展）
- ✅ **GPU加速训练**：完全支持CUDA加速
- ✅ **灵活的配置系统**：基于YAML的配置文件，易于调整参数
- ✅ **全面的评估指标**：MAE、RMSE、MAPE、R²、地理距离误差等
- ✅ **鲁棒性测试**：噪声干扰下的性能评估
- ✅ **可视化工具**：训练历史、预测结果、轨迹对比等
- ✅ **生产就绪**：包含训练、预测、评估完整流程

---

## 🏗️ 项目结构

```
time_series_prediction/
├── config/
│   └── config.yaml              # 配置文件
├── data/
│   ├── raw/                     # 原始数据目录
│   │   └── ship_data.csv        # 船舶AIS数据
│   └── processed/               # 处理后数据
│       └── preprocessor.pkl     # 预处理器（训练后生成）
├── models/
│   ├── __init__.py
│   ├── base_model.py            # 基础模型类
│   ├── lstm_model.py            # LSTM/GRU模型
│   └── transformer_model.py     # Transformer模型
├── preprocessing/
│   ├── __init__.py
│   ├── data_loader.py           # 数据加载器
│   ├── preprocessor.py          # 数据预处理器
│   └── feature_engineer.py      # 特征工程
├── training/
│   ├── __init__.py
│   ├── trainer.py               # 训练器
│   └── evaluator.py             # 评估器
├── utils/
│   ├── __init__.py
│   ├── metrics.py               # 评估指标
│   └── visualization.py         # 可视化工具
├── checkpoints/                 # 模型检查点目录
├── logs/                        # 日志目录
├── runs/                        # TensorBoard日志
├── results/                     # 结果输出目录
├── train.py                     # 训练脚本
├── predict.py                   # 预测脚本
├── generate_sample_data.py      # 示例数据生成脚本
├── requirements.txt             # 依赖包
└── README.md                    # 项目文档
```

---

## 🚀 快速开始

### 1. 环境配置

#### 创建虚拟环境（推荐）

```bash
# 使用conda
conda create -n ship_prediction python=3.10
conda activate ship_prediction

# 或使用venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

#### 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

#### 方式一：使用示例数据生成脚本

```bash
python generate_sample_data.py
```

这将生成一个示例船舶AIS数据文件 `data/raw/ship_data.csv`，包含以下字段：
- `longitude`: 经度
- `latitude`: 纬度
- `speed`: 航速（节）
- `course`: 航向（度）

#### 方式二：使用自己的数据

将CSV数据文件放置在 `data/raw/` 目录下，确保包含以下列：
- `longitude`, `latitude`, `speed`, `course`

数据格式示例：

```csv
longitude,latitude,speed,course
121.5000,31.2000,15.5,45.0
121.5010,31.2008,15.3,44.5
121.5020,31.2016,15.8,44.0
...
```

### 3. 配置模型

编辑 `config/config.yaml` 文件，根据需求调整参数：

```yaml
# 选择模型类型
model:
  model_type: "LSTM"  # 可选: LSTM, GRU, Transformer

# 调整训练参数
training:
  batch_size: 64
  num_epochs: 100
  learning_rate: 0.001
  device: "cuda"  # 或 "cpu"
```

### 4. 训练模型

```bash
python train.py --config config/config.yaml
```

训练过程将：
- 自动划分训练/验证/测试集（70%/20%/10%）
- 实时显示训练进度和指标
- 保存最佳模型到 `checkpoints/best_model.pth`
- 生成TensorBoard日志到 `runs/`
- 保存训练历史图到 `results/training_history.png`

#### 查看训练过程（TensorBoard）

```bash
tensorboard --logdir=runs/
```

### 5. 模型评估与预测

```bash
# 从文件预测
python predict.py \
    --config config/config.yaml \
    --model_path checkpoints/best_model.pth \
    --mode file \
    --input_file data/raw/ship_data.csv \
    --output_file results/predictions.csv \
    --visualize
```

预测结果将包含：
- 预测的经纬度坐标
- 评估指标（MAE、RMSE等）
- 可视化图表（`--visualize`选项）

---

## 📊 模型架构

### LSTM模型

```
输入: (batch_size, 6, 4)  # 6个时间步，4个特征
  ↓
LSTM层1 (hidden_size=64)
  ↓
Batch Normalization
  ↓
Dropout (0.2)
  ↓
LSTM层2 (hidden_size=32)
  ↓
Dropout (0.2)
  ↓
全连接层
  ↓
输出: (batch_size, 3, 2)  # 3个时间步，2个目标（经度、纬度）
```

### Transformer模型

```
输入: (batch_size, 6, 4)
  ↓
输入嵌入 (input_dim → d_model)
  ↓
位置编码
  ↓
Transformer编码器 (多层)
  ↓
解码器 (全连接)
  ↓
输出: (batch_size, 3, 2)
```

---

## 🔧 详细使用指南

### 数据预处理

项目实现了完整的数据预处理流程：

1. **异常值处理**
   - 航速限制：0-30节（可配置）
   - 经度：-180°至180°
   - 纬度：-90°至90°

2. **缺失值填充**
   - `forward`：前向填充
   - `interpolate`：三次插值（推荐）

3. **标准化方法**
   - `zscore`：Z-score标准化（推荐）
   - `minmax`：Min-Max归一化

### 模型配置

#### LSTM配置示例

```yaml
model:
  model_type: "LSTM"
  lstm:
    hidden_size: 64      # 隐藏层大小
    num_layers: 2        # LSTM层数
    dropout: 0.2         # Dropout比例
    bidirectional: false # 是否双向
```

#### Transformer配置示例

```yaml
model:
  model_type: "Transformer"
  transformer:
    d_model: 64          # 模型维度
    nhead: 4             # 注意力头数
    num_layers: 2        # Transformer层数
    dim_feedforward: 256 # 前馈网络维度
    dropout: 0.2
```

### 训练配置

```yaml
training:
  batch_size: 64           # 批次大小
  num_epochs: 100          # 训练轮数
  learning_rate: 0.001     # 学习率
  weight_decay: 0.001      # L2正则化
  
  # 优化器选择
  optimizer: "adam"        # adam, adamw, sgd
  
  # 学习率调度器
  scheduler:
    type: "ReduceLROnPlateau"
    patience: 10
    factor: 0.5
  
  # 早停配置
  early_stopping:
    patience: 20
    min_delta: 0.0001
  
  # 损失函数
  loss_function: "weighted_mse"  # mse, mae, huber, weighted_mse
  time_step_weights: [1.0, 0.8, 0.6]  # 时间步权重（近期权重高）
```

### 评估指标说明

| 指标 | 说明 | 目标值 |
|------|------|--------|
| **MAE** | 平均绝对误差 | < 0.005度（约550米） |
| **RMSE** | 均方根误差 | < 0.01度（约1.1公里） |
| **MAPE** | 平均绝对百分比误差 | < 5% |
| **R²** | 决定系数 | > 0.95 |
| **Distance Error** | 地理距离误差（km） | < 1.0公里 |

---

## 📈 高级功能

### 1. 鲁棒性测试

评估模型在噪声干扰下的性能：

```python
# 在config.yaml中启用
evaluation:
  robustness_test:
    enabled: true
    noise_level: 0.5  # 航速噪声±0.5节
```

### 2. 按时间步评估

分析不同预测步长的误差变化：

```bash
# 训练脚本会自动进行按时间步评估
python train.py
```

### 3. 推理时间测量

评估模型的实时性能：

- 单样本推理时间
- 批量推理吞吐量

目标：单样本推理 < 10ms

### 4. 自定义损失函数

实现了加权MSE损失，近期时间步权重更高：

```yaml
training:
  loss_function: "weighted_mse"
  time_step_weights: [1.0, 0.8, 0.6]  # 第1、2、3步的权重
```

---

## 🎯 模型选择建议

| 场景 | 推荐模型 | 原因 |
|------|----------|------|
| **短期预测**（< 30分钟） | LSTM | 训练快、效果好、资源占用小 |
| **长期预测**（> 1小时） | Transformer | 更好的长序列建模能力 |
| **实时推理** | GRU | 比LSTM更快，性能相近 |
| **小样本数据** | LSTM | 参数少，不易过拟合 |
| **大规模数据** | Transformer | 可并行训练，充分利用数据 |

---

## 💡 扩展开发

### 添加新模型

1. 在 `models/` 目录创建新模型文件
2. 继承 `BaseTimeSeriesModel` 类
3. 实现 `forward` 方法
4. 在 `train.py` 和 `predict.py` 中注册模型

示例：

```python
# models/my_model.py
from models.base_model import BaseTimeSeriesModel

class MyModel(BaseTimeSeriesModel):
    def __init__(self, input_dim, output_dim, lookback_window, prediction_window):
        super().__init__(input_dim, output_dim, lookback_window, prediction_window)
        # 定义模型结构
        
    def forward(self, x):
        # 实现前向传播
        return output
```

### 添加新的评估指标

在 `utils/metrics.py` 中添加新函数：

```python
def calculate_new_metric(y_true, y_pred):
    # 实现新指标
    return metric_value
```

### 自定义数据处理

修改 `preprocessing/preprocessor.py` 中的方法或添加新方法。

---

## 🐛 常见问题

### Q1: CUDA out of memory

**解决方案**：
- 减小 `batch_size`（如从64降到32）
- 减小模型大小（`hidden_size`）
- 使用梯度累积

### Q2: 训练损失不下降

**解决方案**：
- 检查学习率（尝试0.0001）
- 检查数据标准化是否正确
- 增加训练轮数
- 尝试其他优化器（AdamW）

### Q3: 过拟合问题

**解决方案**：
- 增加Dropout比例（0.3-0.5）
- 增加L2正则化（weight_decay）
- 使用数据增强
- 减少模型复杂度

### Q4: 预测结果偏差大

**解决方案**：
- 检查数据质量和异常值
- 增加训练数据量
- 调整预测窗口长度
- 尝试不同的模型架构

### Q5: 中文字体警告 (Glyph missing from font)

**现象**：
```
UserWarning: Glyph 35757 (\N{CJK UNIFIED IDEOGRAPH-8BAD}) missing from font(s) DejaVu Sans.
```

**原因**：matplotlib 默认字体不支持中文字符

**解决方案**：
- ✅ **已自动配置**：项目已在 `utils/visualization.py` 中自动配置中文字体
- 警告不影响使用，图片中的中文会正常显示
- 如需完全消除警告，运行测试脚本验证：
  ```bash
  python test_chinese_font.py
  ```
- 详细说明请查看：`docs/FONT_SETUP.md`

**Windows用户**：系统自带微软雅黑字体，无需额外配置

**Linux用户**：如需安装中文字体
```bash
# Ubuntu/Debian
sudo apt-get install fonts-wqy-microhei

# CentOS/RHEL  
sudo yum install wqy-microhei-fonts
```

---

## 📚 参考资料

### 相关论文

1. **LSTM**: Hochreiter & Schmidhuber (1997). Long Short-Term Memory
2. **Transformer**: Vaswani et al. (2017). Attention Is All You Need
3. **Time Series Forecasting**: Lim et al. (2021). Temporal Fusion Transformers

### 技术文档

- [PyTorch官方文档](https://pytorch.org/docs/)
- [时序预测最佳实践](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

---

## 📝 更新日志

### v1.0.0 (2025-10-24)

- ✅ 初始版本发布
- ✅ 支持LSTM、GRU、Transformer模型
- ✅ 完整的训练和预测流程
- ✅ 全面的评估和可视化工具

---

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

---

## 📄 许可证

本项目采用MIT许可证 - 详见LICENSE文件

---

## 👥 作者

- 项目开发：Chuanguang Zhu
- 技术支持：欢迎提交Issue

---

## 🙏 致谢

感谢以下开源项目的支持：

- PyTorch
- NumPy & Pandas
- Matplotlib
- TensorBoard

---

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- 提交Issue: [GitHub Issues](https://github.com/ZGuangJie/time_series_prediction/issues)
- 邮件: Guangjie98@outlook.com

---

**Happy Predicting! 🚢⚓**

