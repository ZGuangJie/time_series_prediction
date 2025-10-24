"""
训练器模块
负责模型训练、验证和保存
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, Optional
import os
import logging
from tqdm import tqdm

from utils.metrics import calculate_mae, calculate_rmse

logger = logging.getLogger(__name__)


class WeightedMSELoss(nn.Module):
    """加权MSE损失（近期时间步权重更高）"""
    
    def __init__(self, weights: Optional[list] = None):
        """
        初始化加权MSE损失
        
        Args:
            weights: 时间步权重列表
        """
        super(WeightedMSELoss, self).__init__()
        self.weights = weights
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算加权MSE损失
        
        Args:
            pred: 预测值 (batch_size, prediction_window, output_dim)
            target: 目标值 (batch_size, prediction_window, output_dim)
            
        Returns:
            加权MSE损失
        """
        if self.weights is None:
            return nn.functional.mse_loss(pred, target)
        
        # 将权重转换为张量
        weights = torch.tensor(self.weights, device=pred.device).view(1, -1, 1)
        
        # 计算加权MSE
        squared_errors = (pred - target) ** 2
        weighted_errors = squared_errors * weights
        
        return weighted_errors.mean()


class Trainer:
    """模型训练器"""
    
    def __init__(self, 
                 model: nn.Module,
                 config: dict,
                 device: str = 'cuda'):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            config: 训练配置字典
            device: 训练设备
        """
        self.model = model
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 设置优化器
        self.optimizer = self._setup_optimizer()
        
        # 设置损失函数
        self.criterion = self._setup_loss_function()
        
        # 设置学习率调度器
        self.scheduler = self._setup_scheduler()
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'train_rmse': [],
            'val_rmse': []
        }
        
        # TensorBoard
        self.writer = SummaryWriter(config.get('logging', {}).get('tensorboard_dir', 'runs/'))
        
        # 早停
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        logger.info(f"训练器初始化完成 - 设备: {self.device}")
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """设置优化器"""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 0.001)
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            logger.warning(f"未知的优化器 '{optimizer_name}'，使用Adam")
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def _setup_loss_function(self) -> nn.Module:
        """设置损失函数"""
        loss_name = self.config.get('loss_function', 'mse').lower()
        
        if loss_name == 'mse':
            return nn.MSELoss()
        elif loss_name == 'mae':
            return nn.L1Loss()
        elif loss_name == 'huber':
            return nn.HuberLoss()
        elif loss_name == 'weighted_mse':
            weights = self.config.get('time_step_weights', None)
            return WeightedMSELoss(weights)
        else:
            logger.warning(f"未知的损失函数 '{loss_name}'，使用MSE")
            return nn.MSELoss()
    
    def _setup_scheduler(self) -> Optional[object]:
        """设置学习率调度器"""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'ReduceLROnPlateau')
        
        if scheduler_type == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=scheduler_config.get('patience', 10),
                factor=scheduler_config.get('factor', 0.5)
            )
        elif scheduler_type == 'StepLR':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 10),
                gamma=scheduler_config.get('gamma', 0.5)
            )
        elif scheduler_type == 'CosineAnnealingLR':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', 50)
            )
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            包含训练指标的字典
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc='训练中')
        for batch_X, batch_y in pbar:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            
            # 计算损失
            loss = self.criterion(outputs, batch_y)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            all_preds.append(outputs.detach().cpu().numpy())
            all_targets.append(batch_y.detach().cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        
        # 计算评估指标
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        mae = calculate_mae(all_targets, all_preds)
        rmse = calculate_rmse(all_targets, all_preds)
        
        return {
            'loss': avg_loss,
            'mae': mae,
            'rmse': rmse
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            包含验证指标的字典
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        mae = calculate_mae(all_targets, all_preds)
        rmse = calculate_rmse(all_targets, all_preds)
        
        return {
            'loss': avg_loss,
            'mae': mae,
            'rmse': rmse
        }
    
    def train(self, 
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray):
        """
        完整的训练流程
        
        Args:
            X_train: 训练输入数据
            y_train: 训练目标数据
            X_val: 验证输入数据
            y_val: 验证目标数据
        """
        # 创建数据加载器
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        batch_size = self.config.get('batch_size', 64)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        num_epochs = self.config.get('num_epochs', 100)
        early_stopping_config = self.config.get('early_stopping', {})
        patience = early_stopping_config.get('patience', 20)
        
        logger.info(f"开始训练 - Epochs: {num_epochs}, Batch Size: {batch_size}")
        
        for epoch in range(num_epochs):
            # 训练
            train_metrics = self.train_epoch(train_loader)
            
            # 验证
            val_metrics = self.validate(val_loader)
            
            # 记录历史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_mae'].append(train_metrics['mae'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['train_rmse'].append(train_metrics['rmse'])
            self.history['val_rmse'].append(val_metrics['rmse'])
            
            # TensorBoard记录
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('MAE/train', train_metrics['mae'], epoch)
            self.writer.add_scalar('MAE/val', val_metrics['mae'], epoch)
            
            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # 打印进度
            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val MAE: {val_metrics['mae']:.4f}, "
                f"Val RMSE: {val_metrics['rmse']:.4f}"
            )
            
            # 保存最佳模型
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.save_checkpoint('best')
            else:
                self.patience_counter += 1
            
            # 早停
            if self.patience_counter >= patience:
                logger.info(f"早停触发 - 在Epoch {epoch+1}")
                break
            
            # 定期保存
            save_every_n = self.config.get('logging', {}).get('save_every_n_epochs', 10)
            if (epoch + 1) % save_every_n == 0:
                self.save_checkpoint(f'epoch_{epoch+1}')
        
        self.writer.close()
        logger.info("训练完成")
    
    def save_checkpoint(self, name: str = 'checkpoint'):
        """
        保存模型检查点
        
        Args:
            name: 检查点名称
        """
        checkpoint_dir = self.config.get('logging', {}).get('checkpoint_dir', 'checkpoints/')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        path = os.path.join(checkpoint_dir, f'{name}_model.pth')
        self.model.save_model(path)
        
        # 同时保存训练历史
        history_path = os.path.join(checkpoint_dir, f'{name}_history.npy')
        np.save(history_path, self.history)
        
        logger.info(f"检查点已保存: {path}")

