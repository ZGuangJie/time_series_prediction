"""
数据加载模块
负责从CSV文件加载船舶AIS数据
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ShipDataLoader:
    """船舶数据加载器"""
    
    def __init__(self, file_path: str):
        """
        初始化数据加载器
        
        Args:
            file_path: CSV数据文件路径
        """
        self.file_path = file_path
        self.data = None
        
    def load_data(self, 
                  required_columns: Optional[list] = None) -> pd.DataFrame:
        """
        加载CSV数据
        
        Args:
            required_columns: 必需的列名列表
            
        Returns:
            加载的DataFrame
        """
        try:
            logger.info(f"正在从 {self.file_path} 加载数据...")
            self.data = pd.read_csv(self.file_path)
            logger.info(f"成功加载 {len(self.data)} 条记录")
            
            # 检查必需列
            if required_columns:
                missing_cols = set(required_columns) - set(self.data.columns)
                if missing_cols:
                    raise ValueError(f"缺少必需的列: {missing_cols}")
            
            return self.data
            
        except FileNotFoundError:
            logger.error(f"文件未找到: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"加载数据时出错: {str(e)}")
            raise
    
    def get_data_info(self) -> dict:
        """
        获取数据基本信息
        
        Returns:
            包含数据统计信息的字典
        """
        if self.data is None:
            raise ValueError("数据尚未加载，请先调用 load_data()")
        
        info = {
            'num_records': len(self.data),
            'num_features': len(self.data.columns),
            'columns': list(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'data_types': self.data.dtypes.to_dict(),
            'statistics': self.data.describe().to_dict()
        }
        
        return info
    
    def split_data(self, 
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.2,
                   test_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        按时间顺序划分数据集
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            
        Returns:
            (训练集, 验证集, 测试集)
        """
        if self.data is None:
            raise ValueError("数据尚未加载，请先调用 load_data()")
        
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("训练、验证和测试集比例之和必须为1")
        
        n = len(self.data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data = self.data.iloc[:train_end].copy()
        val_data = self.data.iloc[train_end:val_end].copy()
        test_data = self.data.iloc[val_end:].copy()
        
        logger.info(f"数据划分完成 - 训练集: {len(train_data)}, "
                   f"验证集: {len(val_data)}, 测试集: {len(test_data)}")
        
        return train_data, val_data, test_data

