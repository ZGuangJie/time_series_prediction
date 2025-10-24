"""
预测脚本
用于使用训练好的模型进行船舶位置预测
"""
import yaml
import argparse
import logging
import sys
import numpy as np
import pandas as pd
import torch
import pickle
import os

from models.lstm_model import LSTMModel, GRUModel
from models.transformer_model import TransformerModel
from training.evaluator import Evaluator
from utils.visualization import plot_predictions, plot_trajectory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
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
        lstm_config = model_config['lstm']
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
    
    return model


def preprocess_input(data: pd.DataFrame,
                    preprocessor,
                    config: dict) -> np.ndarray:
    """
    预处理输入数据
    
    Args:
        data: 原始数据
        preprocessor: 预处理器
        config: 配置字典
        
    Returns:
        预处理后的数组
    """
    # 提取特征
    features = config['data']['input_features']
    data_array = data[features].values
    
    # 标准化
    data_normalized = np.zeros_like(data_array)
    for i, feature in enumerate(features):
        data_normalized[:, i] = preprocessor.inverse_normalize(
            data_array[:, i], feature
        )
    
    return data_normalized


def postprocess_output(predictions: np.ndarray,
                      preprocessor,
                      config: dict) -> np.ndarray:
    """
    后处理预测输出（逆标准化）
    
    Args:
        predictions: 标准化的预测值
        preprocessor: 预处理器
        config: 配置字典
        
    Returns:
        原始尺度的预测值
    """
    target_features = config['data']['target_features']
    
    # 逆标准化
    predictions_denorm = np.zeros_like(predictions)
    for i, feature in enumerate(target_features):
        predictions_denorm[:, :, i] = preprocessor.inverse_normalize(
            predictions[:, :, i], feature
        )
    
    return predictions_denorm


def predict_from_file(args, config):
    """从文件读取数据并预测"""
    logger.info("=" * 50)
    logger.info("从文件预测")
    logger.info("=" * 50)
    
    # 加载模型
    logger.info(f"加载模型: {args.model_path}")
    model = create_model(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_model(args.model_path, device=device)
    
    # 加载预处理器
    preprocessor_path = os.path.join(
        config['data']['processed_data_path'],
        'preprocessor.pkl'
    )
    logger.info(f"加载预处理器: {preprocessor_path}")
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    # 加载数据
    logger.info(f"加载输入数据: {args.input_file}")
    data = pd.read_csv(args.input_file)
    
    # 预处理
    from preprocessing.preprocessor import DataPreprocessor
    from preprocessing.feature_engineer import FeatureEngineer
    
    temp_preprocessor = DataPreprocessor(config['preprocessing'])
    temp_preprocessor.scaler_params = preprocessor.scaler_params
    
    data_processed = temp_preprocessor.preprocess_pipeline(
        data,
        features=config['data']['input_features'] + config['data']['target_features'],
        fit=False
    )
    
    # 创建序列
    feature_engineer = FeatureEngineer(
        lookback_window=config['data']['lookback_window'],
        prediction_window=config['data']['prediction_window'],
        input_features=config['data']['input_features'],
        target_features=config['data']['target_features']
    )
    
    X, y = feature_engineer.create_sequences(data_processed)
    
    # 预测
    evaluator = Evaluator(model, device=device)
    predictions = evaluator.predict(X)
    
    # 逆标准化
    predictions_denorm = np.zeros_like(predictions)
    y_denorm = np.zeros_like(y)
    
    for i, feature in enumerate(config['data']['target_features']):
        predictions_denorm[:, :, i] = preprocessor.inverse_normalize(
            predictions[:, :, i], feature
        )
        y_denorm[:, :, i] = preprocessor.inverse_normalize(
            y[:, :, i], feature
        )
    
    # 评估
    from utils.metrics import evaluate_all_metrics
    metrics = evaluate_all_metrics(y_denorm, predictions_denorm)
    
    logger.info("=" * 50)
    logger.info("预测结果:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.6f}")
    logger.info("=" * 50)
    
    # 可视化
    if args.visualize:
        logger.info("生成可视化结果...")
        plot_predictions(
            y_denorm,
            predictions_denorm,
            num_samples=5,
            save_path='results/predictions_comparison.png'
        )
        
        # 绘制轨迹
        sample_idx = 0
        plot_trajectory(
            y_denorm[sample_idx, :, 0],  # 经度
            y_denorm[sample_idx, :, 1],  # 纬度
            predictions_denorm[sample_idx, :, 0],
            predictions_denorm[sample_idx, :, 1],
            save_path='results/trajectory_comparison.png'
        )
    
    # 保存预测结果
    if args.output_file:
        logger.info(f"保存预测结果: {args.output_file}")
        
        # 将结果转换为DataFrame
        results_list = []
        for i in range(len(predictions_denorm)):
            for t in range(predictions_denorm.shape[1]):
                results_list.append({
                    'sample_id': i,
                    'time_step': t + 1,
                    'true_longitude': y_denorm[i, t, 0],
                    'true_latitude': y_denorm[i, t, 1],
                    'pred_longitude': predictions_denorm[i, t, 0],
                    'pred_latitude': predictions_denorm[i, t, 1]
                })
        
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(args.output_file, index=False)
        logger.info("预测结果已保存")


def predict_realtime(args, config):
    """实时预测（从最新数据）"""
    logger.info("=" * 50)
    logger.info("实时预测模式")
    logger.info("=" * 50)
    
    # 加载模型
    logger.info(f"加载模型: {args.model_path}")
    model = create_model(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_model(args.model_path, device=device)
    
    # 加载预处理器
    preprocessor_path = os.path.join(
        config['data']['processed_data_path'],
        'preprocessor.pkl'
    )
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    # 实时数据输入（示例）
    logger.info("输入最新的船舶数据（过去6个时间步）:")
    logger.info("格式: 经度,纬度,航速,航向")
    
    # 这里可以实现实时数据输入逻辑
    # 例如从传感器、API或用户输入获取数据
    
    logger.info("实时预测功能待实现...")


def main(args):
    """主预测流程"""
    # 加载配置
    config = load_config(args.config)
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    if args.mode == 'file':
        predict_from_file(args, config)
    elif args.mode == 'realtime':
        predict_realtime(args, config)
    else:
        raise ValueError(f"不支持的预测模式: {args.mode}")
    
    logger.info("预测完成!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='船舶位置预测')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='checkpoints/best_model.pth',
        help='模型文件路径'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['file', 'realtime'],
        default='file',
        help='预测模式: file（从文件）或 realtime（实时）'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        help='输入数据文件路径（CSV格式）'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='results/predictions.csv',
        help='预测结果保存路径'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='是否生成可视化结果'
    )
    
    args = parser.parse_args()
    main(args)

