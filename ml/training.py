import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split, KFold
import logging
import pandas as pd
import os
from ml.preprocessing import clean_text
from ml.feature_extraction import extract_features
from ml.model import SpamClassifier, SMSLSTM, SMSCNN

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)

class SMSDataset(Dataset):
    """SMS数据集类，用于PyTorch DataLoader"""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor([self.labels[idx]])

def load_data(filepath):
    """
    加载SMS数据集
    
    参数:
        filepath: CSV文件路径
    
    返回:
        features: 特征矩阵
        labels: 标签向量
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(filepath)
        
        # 确保必要的列存在
        required_columns = ['text', 'label']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("CSV文件必须包含'text'和'label'列")
        
        # 提取特征和标签
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        # 转换标签为数值
        labels = [1 if label == 'spam' else 0 for label in labels]
        
        # 添加元数据列（如果存在）
        send_freq = df.get('send_freq', [0] * len(texts)).tolist()
        is_night = df.get('is_night', [0] * len(texts)).tolist()
        
        # 提取特征
        features = []
        for i, text in enumerate(texts):
            # 清理文本
            cleaned_text = clean_text(text)
            # 提取特征
            feature = extract_features(cleaned_text, None, send_freq[i], is_night[i])
            features.append(feature)
        
        return np.array(features), np.array(labels)
    
    except Exception as e:
        logging.error(f"加载数据错误: {str(e)}")
        return None, None

def train_model(model, features, labels, model_type, model_save_path, epochs=10, batch_size=32, learning_rate=0.001):
    """
    训练模型
    
    参数:
        model: 未训练的模型
        features: 特征矩阵
        labels: 标签向量
        model_type: 模型类型
        model_save_path: 模型保存路径
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
    
    返回:
        model: 训练好的模型
        metrics: 评估指标
    """
    try:
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        
        # 创建数据集和数据加载器
        train_dataset = SMSDataset(X_train, y_train)
        test_dataset = SMSDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # 设置优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        
        # 训练循环
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            
            for features, labels in train_loader:
                # 前向传播
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 每轮结束打印损失
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}")
            
            # 每轮都评估模型
            model.eval()
            with torch.no_grad():
                y_pred = []
                y_true = []
                
                for features, labels in test_loader:
                    outputs = model(features)
                    preds = torch.sigmoid(outputs) > 0.5
                    y_pred.extend(preds.cpu().numpy())
                    y_true.extend(labels.cpu().numpy())
                
                # 计算指标
                accuracy = metrics.accuracy_score(y_true, y_pred)
                precision = metrics.precision_score(y_true, y_pred)
                recall = metrics.recall_score(y_true, y_pred)
                f1 = metrics.f1_score(y_true, y_pred)
                
                logging.info(f"Evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # 创建保存目录（如果不存在）
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # 保存模型
        torch.save(model.state_dict(), model_save_path)
        logging.info(f"模型已保存到: {model_save_path}")
        
        # 返回最终评估指标
        final_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return model, final_metrics
    
    except Exception as e:
        logging.error(f"训练模型错误: {str(e)}")
        return model, {}

def cross_validate(model_class, features, labels, model_type, n_folds=5, epochs=5, batch_size=32, learning_rate=0.001):
    """
    使用K折交叉验证评估模型
    
    参数:
        model_class: 模型类
        features: 特征矩阵
        labels: 标签向量
        model_type: 模型类型
        n_folds: 折数
        epochs: 每折训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
    
    返回:
        metrics: 平均评估指标
    """
    try:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        all_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(features)):
            logging.info(f"开始第 {fold+1} 折...")
            
            # 获取训练集和测试集
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            
            # 创建模型实例
            if model_type in ['roberta', 'bert']:
                model = model_class(input_dim=770)  # 768 + 2 (元数据特征)
            else:
                model = model_class()  # LSTM或CNN模型
            
            # 创建数据集和加载器
            train_dataset = SMSDataset(X_train, y_train)
            test_dataset = SMSDataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            # 设置优化器和损失函数
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.BCEWithLogitsLoss()
            
            # 训练循环
            for epoch in range(epochs):
                model.train()
                train_loss = 0
                
                for batch_features, batch_labels in train_loader:
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # 每轮结束打印损失
                logging.info(f"Fold {fold+1}/{n_folds}, Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}")
            
            # 评估模型
            model.eval()
            with torch.no_grad():
                y_pred = []
                y_true = []
                
                for batch_features, batch_labels in test_loader:
                    outputs = model(batch_features)
                    preds = torch.sigmoid(outputs) > 0.5
                    y_pred.extend(preds.cpu().numpy())
                    y_true.extend(batch_labels.cpu().numpy())
                
                # 计算指标
                accuracy = metrics.accuracy_score(y_true, y_pred)
                precision = metrics.precision_score(y_true, y_pred)
                recall = metrics.recall_score(y_true, y_pred)
                f1 = metrics.f1_score(y_true, y_pred)
                
                logging.info(f"Fold {fold+1} Evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                
                # 添加到所有指标中
                all_metrics['accuracy'].append(accuracy)
                all_metrics['precision'].append(precision)
                all_metrics['recall'].append(recall)
                all_metrics['f1'].append(f1)
        
        # 计算平均指标
        avg_metrics = {
            'accuracy': np.mean(all_metrics['accuracy']),
            'precision': np.mean(all_metrics['precision']),
            'recall': np.mean(all_metrics['recall']),
            'f1': np.mean(all_metrics['f1'])
        }
        
        logging.info(f"交叉验证平均指标 - Accuracy: {avg_metrics['accuracy']:.4f}, Precision: {avg_metrics['precision']:.4f}, Recall: {avg_metrics['recall']:.4f}, F1: {avg_metrics['f1']:.4f}")
        
        return avg_metrics
    
    except Exception as e:
        logging.error(f"交叉验证错误: {str(e)}")
        return {}
