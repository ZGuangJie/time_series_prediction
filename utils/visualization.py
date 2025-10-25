"""
可视化工具模块
提供训练过程和预测结果的可视化功能
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import Dict, List, Optional
import os
import platform

# 配置中文字体支持
def setup_chinese_font():
    """配置matplotlib支持中文显示"""
    system = platform.system()
    
    if system == 'Windows':
        # Windows系统使用微软雅黑或SimHei
        try:
            matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong']
        except:
            matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    elif system == 'Darwin':  # macOS
        matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'STHeiti', 'STSong']
    else:  # Linux
        # matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback']
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']

    # 解决负号'-'显示为方块的问题
    matplotlib.rcParams['axes.unicode_minus'] = False

# 初始化中文字体
setup_chinese_font()


def plot_training_history(history: Dict[str, List[float]], 
                          save_path: Optional[str] = None):
    """
    绘制训练历史曲线
    
    Args:
        history: 训练历史字典（包含loss, val_loss等）
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    if 'train_loss' in history:
        plt.plot(history['train_loss'], label='训练损失', linewidth=2)
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='验证损失', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('训练和验证损失', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制评估指标
    plt.subplot(1, 2, 2)
    metrics_to_plot = ['mae', 'rmse']
    for metric in metrics_to_plot:
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        if train_key in history:
            plt.plot(history[train_key], label=f'训练 {metric.upper()}', linewidth=2)
        if val_key in history:
            plt.plot(history[val_key], label=f'验证 {metric.upper()}', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.title('评估指标', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图已保存到 {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_predictions(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    num_samples: int = 5,
                    save_path: Optional[str] = None):
    """
    绘制预测结果对比
    
    Args:
        y_true: 真实值 (num_samples, prediction_window, output_dim)
        y_pred: 预测值 (num_samples, prediction_window, output_dim)
        num_samples: 显示的样本数量
        save_path: 保存路径（可选）
    """
    num_samples = min(num_samples, y_true.shape[0])
    output_dim = y_true.shape[-1]
    prediction_window = y_true.shape[1]
    
    fig, axes = plt.subplots(num_samples, output_dim, figsize=(12, 3*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    if output_dim == 1:
        axes = axes.reshape(-1, 1)
    
    feature_names = ['经度', '纬度'] if output_dim == 2 else [f'特征{i+1}' for i in range(output_dim)]
    
    for i in range(num_samples):
        for j in range(output_dim):
            ax = axes[i, j]
            
            time_steps = range(prediction_window)
            ax.plot(time_steps, y_true[i, :, j], 'bo-', label='真实值', linewidth=2, markersize=8)
            ax.plot(time_steps, y_pred[i, :, j], 'r^--', label='预测值', linewidth=2, markersize=8)
            
            ax.set_xlabel('时间步', fontsize=10)
            ax.set_ylabel('值', fontsize=10)
            ax.set_title(f'样本{i+1} - {feature_names[j]}', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测结果图已保存到 {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_trajectory(lon_true: np.ndarray,
                   lat_true: np.ndarray,
                   lon_pred: np.ndarray,
                   lat_pred: np.ndarray,
                   save_path: Optional[str] = None):
    """
    绘制船舶轨迹对比
    
    Args:
        lon_true: 真实经度
        lat_true: 真实纬度
        lon_pred: 预测经度
        lat_pred: 预测纬度
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(10, 8))
    
    plt.plot(lon_true, lat_true, 'bo-', label='真实轨迹', linewidth=2, markersize=8)
    plt.plot(lon_pred, lat_pred, 'r^--', label='预测轨迹', linewidth=2, markersize=8)
    
    # 标记起点和终点
    plt.plot(lon_true[0], lat_true[0], 'go', markersize=15, label='起点')
    plt.plot(lon_true[-1], lat_true[-1], 'rs', markersize=15, label='终点')
    
    plt.xlabel('经度', fontsize=12)
    plt.ylabel('纬度', fontsize=12)
    plt.title('船舶轨迹预测对比', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"轨迹图已保存到 {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_error_distribution(errors: np.ndarray, 
                           save_path: Optional[str] = None):
    """
    绘制误差分布
    
    Args:
        errors: 误差数组
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(12, 4))
    
    # 误差直方图
    plt.subplot(1, 2, 1)
    plt.hist(errors.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('误差', fontsize=12)
    plt.ylabel('频数', fontsize=12)
    plt.title('误差分布直方图', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 误差箱线图
    plt.subplot(1, 2, 2)
    plt.boxplot(errors.flatten())
    plt.ylabel('误差', fontsize=12)
    plt.title('误差箱线图', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"误差分布图已保存到 {save_path}")
    else:
        plt.show()
    
    plt.close()

