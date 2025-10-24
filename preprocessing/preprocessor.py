"""
数据预处理模块
负责数据清洗、异常值处理、缺失值填充、标准化等
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import interpolate
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, config: dict):
        """
        初始化预处理器
        
        Args:
            config: 预处理配置字典
        """
        self.config = config
        self.scaler_params = {}  # 存储标准化参数
        
    def remove_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        移除异常值
        
        Args:
            data: 原始数据
            
        Returns:
            清洗后的数据
        """
        anomaly_config = self.config.get('anomaly_detection', {})
        
        # 创建数据副本
        cleaned_data = data.copy()
        initial_count = len(cleaned_data)
        
        # 检查航速异常
        if 'speed' in cleaned_data.columns:
            max_speed = anomaly_config.get('max_speed', 30.0)
            cleaned_data = cleaned_data[
                (cleaned_data['speed'] >= 0) & 
                (cleaned_data['speed'] <= max_speed)
            ]
        
        # 检查纬度异常
        if 'latitude' in cleaned_data.columns:
            min_lat = anomaly_config.get('min_latitude', -90.0)
            max_lat = anomaly_config.get('max_latitude', 90.0)
            cleaned_data = cleaned_data[
                (cleaned_data['latitude'] >= min_lat) & 
                (cleaned_data['latitude'] <= max_lat)
            ]
        
        # 检查经度异常
        if 'longitude' in cleaned_data.columns:
            min_lon = anomaly_config.get('min_longitude', -180.0)
            max_lon = anomaly_config.get('max_longitude', 180.0)
            cleaned_data = cleaned_data[
                (cleaned_data['longitude'] >= min_lon) & 
                (cleaned_data['longitude'] <= max_lon)
            ]
        
        removed_count = initial_count - len(cleaned_data)
        logger.info(f"移除了 {removed_count} 条异常数据 ({removed_count/initial_count*100:.2f}%)")
        
        return cleaned_data.reset_index(drop=True)
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            data: 包含缺失值的数据
            
        Returns:
            填充后的数据
        """
        strategy = self.config.get('missing_value_strategy', 'interpolate')
        data_filled = data.copy()
        
        for column in data_filled.columns:
            missing_count = data_filled[column].isnull().sum()
            if missing_count > 0:
                logger.info(f"列 '{column}' 有 {missing_count} 个缺失值")
                
                if strategy == 'forward':
                    # 前向填充
                    data_filled[column].fillna(method='ffill', inplace=True)
                    data_filled[column].fillna(method='bfill', inplace=True)
                    
                elif strategy == 'interpolate':
                    # 插值填充
                    data_filled[column].interpolate(
                        method='cubic', 
                        limit_direction='both',
                        inplace=True
                    )
                    # 处理边界缺失值
                    data_filled[column].fillna(method='bfill', inplace=True)
                    data_filled[column].fillna(method='ffill', inplace=True)
        
        logger.info(f"使用 '{strategy}' 策略完成缺失值填充")
        return data_filled
    
    def normalize_data(self, 
                       data: pd.DataFrame,
                       features: list,
                       fit: bool = True) -> pd.DataFrame:
        """
        标准化数据
        
        Args:
            data: 原始数据
            features: 需要标准化的特征列表
            fit: 是否计算标准化参数（训练集为True，验证/测试集为False）
            
        Returns:
            标准化后的数据
        """
        method = self.config.get('normalization', 'zscore')
        normalized_data = data.copy()
        
        for feature in features:
            if feature not in data.columns:
                logger.warning(f"特征 '{feature}' 不在数据中，跳过")
                continue
            
            if fit:
                if method == 'zscore':
                    # Z-score标准化
                    mean = data[feature].mean()
                    std = data[feature].std()
                    self.scaler_params[feature] = {'mean': mean, 'std': std}
                    
                elif method == 'minmax':
                    # Min-Max标准化
                    min_val = data[feature].min()
                    max_val = data[feature].max()
                    self.scaler_params[feature] = {'min': min_val, 'max': max_val}
            
            # 应用标准化
            if method == 'zscore':
                params = self.scaler_params[feature]
                normalized_data[feature] = (
                    (data[feature] - params['mean']) / (params['std'] + 1e-8)
                )
                
            elif method == 'minmax':
                params = self.scaler_params[feature]
                range_val = params['max'] - params['min']
                normalized_data[feature] = (
                    (data[feature] - params['min']) / (range_val + 1e-8)
                )
        
        logger.info(f"使用 '{method}' 方法完成数据标准化")
        return normalized_data
    
    def inverse_normalize(self, 
                         data: np.ndarray,
                         feature: str) -> np.ndarray:
        """
        逆标准化（还原数据到原始尺度）
        
        Args:
            data: 标准化后的数据
            feature: 特征名称
            
        Returns:
            原始尺度的数据
        """
        if feature not in self.scaler_params:
            raise ValueError(f"特征 '{feature}' 的标准化参数未找到")
        
        method = self.config.get('normalization', 'zscore')
        params = self.scaler_params[feature]
        
        if method == 'zscore':
            return data * params['std'] + params['mean']
        elif method == 'minmax':
            range_val = params['max'] - params['min']
            return data * range_val + params['min']
        
        return data
    
    def convert_speed_units(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        转换航速单位（节 → 度/小时）
        1节 ≈ 0.01668 度/小时
        
        Args:
            data: 包含speed列的数据
            
        Returns:
            转换后的数据
        """
        if 'speed' in data.columns:
            data_converted = data.copy()
            data_converted['speed'] = data['speed'] * 0.01668
            logger.info("航速单位已转换（节 → 度/小时）")
            return data_converted
        return data
    
    def preprocess_pipeline(self, 
                           data: pd.DataFrame,
                           features: list,
                           fit: bool = True) -> pd.DataFrame:
        """
        完整的预处理流程
        
        Args:
            data: 原始数据
            features: 需要处理的特征列表
            fit: 是否为训练集（需要计算参数）
            
        Returns:
            预处理后的数据
        """
        logger.info("开始数据预处理流程...")
        
        # 1. 移除异常值
        data = self.remove_anomalies(data)
        
        # 2. 处理缺失值
        data = self.handle_missing_values(data)
        
        # 3. 转换航速单位（可选）
        # data = self.convert_speed_units(data)
        
        # 4. 标准化
        data = self.normalize_data(data, features, fit=fit)
        
        logger.info("数据预处理流程完成")
        return data

