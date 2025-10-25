"""
特征工程模块
负责时序特征构造、衍生特征生成、序列窗口构建等
"""
import numpy as np
import pandas as pd
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """特征工程器"""
    
    def __init__(self, 
                 lookback_window: int,
                 prediction_window: int,
                 input_features: list,
                 target_features: list):
        """
        初始化特征工程器
        
        Args:
            lookback_window: 回顾窗口长度
            prediction_window: 预测窗口长度
            input_features: 输入特征列表
            target_features: 目标特征列表
        """
        self.lookback_window = lookback_window
        self.prediction_window = prediction_window
        self.input_features = input_features
        self.target_features = target_features
        
    def create_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        创建衍生特征
        
        Args:
            data: 原始数据
            
        Returns:
            添加衍生特征后的数据
        """
        data_derived = data.copy()
        
        # 航向变化量
        if 'course' in data.columns:
            data_derived['course_change'] = data['course'].diff().fillna(0)
            # 处理航向跨越360度的情况
            data_derived.loc[data_derived['course_change'] > 180, 'course_change'] -= 360
            data_derived.loc[data_derived['course_change'] < -180, 'course_change'] += 360
        
        # 位置增量
        if 'longitude' in data.columns:
            data_derived['delta_lon'] = data['longitude'].diff().fillna(0)
        
        if 'latitude' in data.columns:
            data_derived['delta_lat'] = data['latitude'].diff().fillna(0)
        
        # 航速变化
        if 'speed' in data.columns:
            data_derived['speed_change'] = data['speed'].diff().fillna(0)
        
        logger.info(f"创建了 {len(data_derived.columns) - len(data.columns)} 个衍生特征")
        return data_derived
    
    def create_sequences(self, 
                        data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时序序列（滑动窗口）
        
        Args:
            data: 预处理后的数据
            
        Returns:
            (输入序列X, 目标序列Y)
            X shape: (样本数, lookback_window, 特征数)
            Y shape: (样本数, prediction_window, 目标特征数)
        """
        # 提取特征数据
        input_data = data[self.input_features].values
        target_data = data[self.target_features].values
        
        X, Y = [], []
        
        # 滑动窗口构建序列
        for i in range(len(data) - self.lookback_window - self.prediction_window + 1):
            # 输入序列：过去lookback_window个时间步
            X.append(input_data[i:i + self.lookback_window])
            
            # 目标序列：未来prediction_window个时间步
            Y.append(target_data[
                i + self.lookback_window:
                i + self.lookback_window + self.prediction_window
            ])
        
        X = np.array(X)
        Y = np.array(Y)
        
        logger.info(f"创建序列完成 - X shape: {X.shape}, Y shape: {Y.shape}")
        return X, Y
    
    def create_single_step_sequences(self, 
                                    data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建单步预测序列
        
        Args:
            data: 预处理后的数据
            
        Returns:
            (输入序列X, 目标Y)
            X shape: (样本数, lookback_window, 特征数)
            Y shape: (样本数, 目标特征数)
        """
        input_data = data[self.input_features].values
        target_data = data[self.target_features].values
        
        X, Y = [], []
        
        for i in range(len(data) - self.lookback_window):
            X.append(input_data[i:i + self.lookback_window])
            Y.append(target_data[i + self.lookback_window])
        
        X = np.array(X)
        Y = np.array(Y)
        
        logger.info(f"创建单步序列完成 - X shape: {X.shape}, Y shape: {Y.shape}")
        return X, Y
    
    def prepare_data_for_training(self,
                                  train_data: pd.DataFrame,
                                  val_data: pd.DataFrame,
                                  test_data: pd.DataFrame) -> dict:
        """
        准备训练、验证和测试数据
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
            test_data: 测试数据
            
        Returns:
            包含所有数据集的字典
        """
        logger.info("准备训练数据...")
        
        # 创建序列
        X_train, y_train = self.create_sequences(train_data)
        X_val, y_val = self.create_sequences(val_data)
        X_test, y_test = self.create_sequences(test_data)
        
        data_dict = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
        
        logger.info("数据准备完成")
        return data_dict

