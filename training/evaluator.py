"""
评估器模块
负责模型评估和鲁棒性测试
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple
import logging

from utils.metrics import evaluate_all_metrics, calculate_distance_error

logger = logging.getLogger(__name__)


class Evaluator:
    """模型评估器"""
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda'):
        """
        初始化评估器
        
        Args:
            model: 要评估的模型
            device: 评估设备
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"评估器初始化完成 - 设备: {self.device}")
    
    def predict(self, 
                X: np.ndarray,
                batch_size: int = 64) -> np.ndarray:
        """
        批量预测
        
        Args:
            X: 输入数据
            batch_size: 批次大小
            
        Returns:
            预测结果
        """
        dataset = TensorDataset(torch.FloatTensor(X))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        
        with torch.no_grad():
            for (batch_X,) in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def evaluate(self, 
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 batch_size: int = 64) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            X_test: 测试输入数据
            y_test: 测试目标数据
            batch_size: 批次大小
            
        Returns:
            评估指标字典
        """
        logger.info("开始模型评估...")
        
        # 预测
        y_pred = self.predict(X_test, batch_size)
        
        # 计算所有指标
        metrics = evaluate_all_metrics(y_test, y_pred)
        
        # 打印评估结果
        logger.info("=" * 50)
        logger.info("评估结果:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name}: {metric_value:.6f}")
        logger.info("=" * 50)
        
        return metrics
    
    def test_robustness(self,
                       X_test: np.ndarray,
                       y_test: np.ndarray,
                       noise_level: float = 0.5,
                       batch_size: int = 64) -> Dict[str, float]:
        """
        鲁棒性测试（加入噪声后的性能）
        
        Args:
            X_test: 测试输入数据
            y_test: 测试目标数据
            noise_level: 噪声水平（航速噪声，单位：节）
            batch_size: 批次大小
            
        Returns:
            包含原始和噪声下指标的字典
        """
        logger.info(f"开始鲁棒性测试 - 噪声水平: ±{noise_level}节")
        
        # 原始性能
        metrics_original = self.evaluate(X_test, y_test, batch_size)
        
        # 添加噪声（假设航速是第3个特征，索引为2）
        X_test_noisy = X_test.copy()
        speed_noise = np.random.normal(0, noise_level, X_test_noisy[:, :, 2].shape)
        X_test_noisy[:, :, 2] += speed_noise
        
        # 噪声下的性能
        y_pred_noisy = self.predict(X_test_noisy, batch_size)
        metrics_noisy = evaluate_all_metrics(y_test, y_pred_noisy)
        
        # 计算性能变化率
        robustness_results = {
            'original_mae': metrics_original['MAE'],
            'noisy_mae': metrics_noisy['MAE'],
            'mae_change_rate': (metrics_noisy['MAE'] - metrics_original['MAE']) / metrics_original['MAE'] * 100,
            'original_rmse': metrics_original['RMSE'],
            'noisy_rmse': metrics_noisy['RMSE'],
            'rmse_change_rate': (metrics_noisy['RMSE'] - metrics_original['RMSE']) / metrics_original['RMSE'] * 100
        }
        
        logger.info("=" * 50)
        logger.info("鲁棒性测试结果:")
        logger.info(f"原始MAE: {robustness_results['original_mae']:.6f}")
        logger.info(f"噪声MAE: {robustness_results['noisy_mae']:.6f}")
        logger.info(f"MAE变化率: {robustness_results['mae_change_rate']:.2f}%")
        logger.info("=" * 50)
        
        return robustness_results
    
    def evaluate_by_time_step(self,
                             X_test: np.ndarray,
                             y_test: np.ndarray,
                             batch_size: int = 64) -> Dict[int, Dict[str, float]]:
        """
        按时间步评估（分析不同预测步长的误差）
        
        Args:
            X_test: 测试输入数据
            y_test: 测试目标数据
            batch_size: 批次大小
            
        Returns:
            每个时间步的评估指标字典
        """
        logger.info("开始按时间步评估...")
        
        y_pred = self.predict(X_test, batch_size)
        prediction_window = y_test.shape[1]
        
        time_step_metrics = {}
        
        for t in range(prediction_window):
            y_true_t = y_test[:, t, :]
            y_pred_t = y_pred[:, t, :]
            
            metrics_t = evaluate_all_metrics(y_true_t, y_pred_t)
            time_step_metrics[t+1] = metrics_t
            
            logger.info(f"时间步 {t+1}: MAE={metrics_t['MAE']:.6f}, RMSE={metrics_t['RMSE']:.6f}")
        
        return time_step_metrics
    
    def analyze_error_distribution(self,
                                  X_test: np.ndarray,
                                  y_test: np.ndarray,
                                  batch_size: int = 64) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        分析误差分布
        
        Args:
            X_test: 测试输入数据
            y_test: 测试目标数据
            batch_size: 批次大小
            
        Returns:
            (误差数组, 误差统计字典)
        """
        y_pred = self.predict(X_test, batch_size)
        errors = np.abs(y_test - y_pred)
        
        error_stats = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'median_error': np.median(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'percentile_95': np.percentile(errors, 95),
            'percentile_99': np.percentile(errors, 99)
        }
        
        logger.info("=" * 50)
        logger.info("误差分布统计:")
        for stat_name, stat_value in error_stats.items():
            logger.info(f"{stat_name}: {stat_value:.6f}")
        logger.info("=" * 50)
        
        return errors, error_stats
    
    def measure_inference_time(self,
                              X_test: np.ndarray,
                              num_iterations: int = 100) -> Dict[str, float]:
        """
        测量推理时间
        
        Args:
            X_test: 测试输入数据
            num_iterations: 测试迭代次数
            
        Returns:
            推理时间统计字典
        """
        import time
        
        # 预热
        _ = self.predict(X_test[:10], batch_size=10)
        
        # 单样本推理时间
        single_times = []
        for _ in range(num_iterations):
            sample = X_test[0:1]
            start = time.time()
            with torch.no_grad():
                _ = self.model(torch.FloatTensor(sample).to(self.device))
            end = time.time()
            single_times.append((end - start) * 1000)  # 转换为毫秒
        
        # 批量推理时间
        batch_size = 64
        batch_times = []
        for _ in range(num_iterations):
            batch = X_test[:batch_size]
            start = time.time()
            _ = self.predict(batch, batch_size=batch_size)
            end = time.time()
            batch_times.append((end - start) * 1000)
        
        time_stats = {
            'single_sample_mean_ms': np.mean(single_times),
            'single_sample_std_ms': np.std(single_times),
            'batch_mean_ms': np.mean(batch_times),
            'batch_std_ms': np.std(batch_times),
            'throughput_samples_per_sec': batch_size / (np.mean(batch_times) / 1000)
        }
        
        logger.info("=" * 50)
        logger.info("推理时间统计:")
        logger.info(f"单样本推理: {time_stats['single_sample_mean_ms']:.2f}±{time_stats['single_sample_std_ms']:.2f} ms")
        logger.info(f"批量推理: {time_stats['batch_mean_ms']:.2f}±{time_stats['batch_std_ms']:.2f} ms")
        logger.info(f"吞吐量: {time_stats['throughput_samples_per_sec']:.2f} 样本/秒")
        logger.info("=" * 50)
        
        return time_stats

