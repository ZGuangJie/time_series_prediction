"""
训练脚本
用于训练船舶位置预测模型
"""
import yaml
import argparse
import logging
import os
import sys
import numpy as np
import torch

from preprocessing.data_loader import ShipDataLoader
from preprocessing.preprocessor import DataPreprocessor
from preprocessing.feature_engineer import FeatureEngineer
from models.lstm_model import LSTMModel, GRUModel
from models.transformer_model import TransformerModel
from training.trainer import Trainer
from training.evaluator import Evaluator
from utils.visualization import plot_training_history

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict):
    """根据配置创建模型"""
    model_config = config['model']
    data_config = config['data']
    
    model_type = model_config['model_type'].upper()
    input_dim = model_config['input_dim']
    output_dim = model_config['output_dim']
    lookback_window = data_config['lookback_window']
    prediction_window = data_config['prediction_window']
    
    if model_type == 'LSTM':
        lstm_config = model_config['lstm']
        model = LSTMModel(
            input_dim=input_dim,
            output_dim=output_dim,
            lookback_window=lookback_window,
            prediction_window=prediction_window,
            hidden_size=lstm_config['hidden_size'],
            num_layers=lstm_config['num_layers'],
            dropout=lstm_config['dropout'],
            bidirectional=lstm_config['bidirectional']
        )
    elif model_type == 'GRU':
        lstm_config = model_config['lstm']  # GRU使用相同配置
        model = GRUModel(
            input_dim=input_dim,
            output_dim=output_dim,
            lookback_window=lookback_window,
            prediction_window=prediction_window,
            hidden_size=lstm_config['hidden_size'],
            num_layers=lstm_config['num_layers'],
            dropout=lstm_config['dropout'],
            bidirectional=lstm_config['bidirectional']
        )
    elif model_type == 'TRANSFORMER':
        transformer_config = model_config['transformer']
        model = TransformerModel(
            input_dim=input_dim,
            output_dim=output_dim,
            lookback_window=lookback_window,
            prediction_window=prediction_window,
            d_model=transformer_config['d_model'],
            nhead=transformer_config['nhead'],
            num_layers=transformer_config['num_layers'],
            dim_feedforward=transformer_config['dim_feedforward'],
            dropout=transformer_config['dropout']
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    logger.info(f"创建模型: {model_type}")
    logger.info(f"模型参数总数: {sum(p.numel() for p in model.parameters())}")
    
    return model


def main(args):
    """主训练流程"""
    # 加载配置
    logger.info(f"加载配置文件: {args.config}")
    config = load_config(args.config)
    
    # 创建必要的目录
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['logging']['tensorboard_dir'], exist_ok=True)
    
    # 1. 数据加载
    logger.info("=" * 50)
    logger.info("步骤 1: 加载数据")
    logger.info("=" * 50)
    
    data_loader = ShipDataLoader(config['data']['raw_data_path'])
    data = data_loader.load_data(
        required_columns=config['data']['input_features']
    )
    
    # 数据划分
    train_data, val_data, test_data = data_loader.split_data(
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio']
    )
    
    # 2. 数据预处理
    logger.info("=" * 50)
    logger.info("步骤 2: 数据预处理")
    logger.info("=" * 50)
    
    preprocessor = DataPreprocessor(config['preprocessing'])
    
    # 训练集预处理（fit=True，计算标准化参数）
    train_data_processed = preprocessor.preprocess_pipeline(
        train_data,
        features=config['data']['input_features'] + config['data']['target_features'],
        fit=True
    )
    
    # 验证集和测试集预处理（fit=False，使用训练集参数）
    val_data_processed = preprocessor.preprocess_pipeline(
        val_data,
        features=config['data']['input_features'] + config['data']['target_features'],
        fit=False
    )
    
    test_data_processed = preprocessor.preprocess_pipeline(
        test_data,
        features=config['data']['input_features'] + config['data']['target_features'],
        fit=False
    )
    
    # 保存预处理器（用于后续预测）
    import pickle
    preprocessor_path = os.path.join(
        config['data']['processed_data_path'],
        'preprocessor.pkl'
    )
    os.makedirs(config['data']['processed_data_path'], exist_ok=True)
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    logger.info(f"预处理器已保存: {preprocessor_path}")
    
    # 3. 特征工程
    logger.info("=" * 50)
    logger.info("步骤 3: 特征工程")
    logger.info("=" * 50)
    
    feature_engineer = FeatureEngineer(
        lookback_window=config['data']['lookback_window'],
        prediction_window=config['data']['prediction_window'],
        input_features=config['data']['input_features'],
        target_features=config['data']['target_features']
    )
    
    # 准备训练数据
    data_dict = feature_engineer.prepare_data_for_training(
        train_data_processed,
        val_data_processed,
        test_data_processed
    )
    
    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    X_val, y_val = data_dict['X_val'], data_dict['y_val']
    X_test, y_test = data_dict['X_test'], data_dict['y_test']
    
    logger.info(f"训练集: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"验证集: X={X_val.shape}, y={y_val.shape}")
    logger.info(f"测试集: X={X_test.shape}, y={y_test.shape}")
    
    # 4. 创建模型
    logger.info("=" * 50)
    logger.info("步骤 4: 创建模型")
    logger.info("=" * 50)
    
    model = create_model(config)
    
    # 5. 训练模型
    logger.info("=" * 50)
    logger.info("步骤 5: 训练模型")
    logger.info("=" * 50)
    
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA不可用，使用CPU训练")
        device = 'cpu'
    
    trainer = Trainer(model, config['training'], device=device)
    trainer.train(X_train, y_train, X_val, y_val)
    
    # 绘制训练历史
    plot_training_history(
        trainer.history,
        save_path='results/training_history.png'
    )
    
    # 6. 评估模型
    logger.info("=" * 50)
    logger.info("步骤 6: 评估模型")
    logger.info("=" * 50)
    
    # 加载最佳模型
    best_model_path = os.path.join(
        config['logging']['checkpoint_dir'],
        'best_model.pth'
    )
    model.load_model(best_model_path, device=device)
    
    evaluator = Evaluator(model, device=device)
    
    # 测试集评估
    test_metrics = evaluator.evaluate(X_test, y_test)
    
    # 鲁棒性测试
    if config['evaluation'].get('robustness_test', {}).get('enabled', False):
        noise_level = config['evaluation']['robustness_test']['noise_level']
        robustness_results = evaluator.test_robustness(X_test, y_test, noise_level)
    
    # 按时间步评估
    time_step_metrics = evaluator.evaluate_by_time_step(X_test, y_test)
    
    # 推理时间测试
    inference_stats = evaluator.measure_inference_time(X_test, num_iterations=100)
    
    # 保存评估结果
    results = {
        'test_metrics': test_metrics,
        'time_step_metrics': time_step_metrics,
        'inference_stats': inference_stats
    }
    
    results_path = 'results/evaluation_results.npy'
    np.save(results_path, results)
    logger.info(f"评估结果已保存: {results_path}")
    
    logger.info("=" * 50)
    logger.info("训练和评估完成!")
    logger.info("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练船舶位置预测模型')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='配置文件路径'
    )
    
    args = parser.parse_args()
    main(args)

