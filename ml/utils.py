import numpy as np
import pandas as pd
import logging
import os
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)

def ensure_directory(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"创建目录: {directory}")

def save_model(model, model_path):
    """
    保存模型 (支持PyTorch和传统机器学习模型)
    
    参数:
        model: 模型实例 (PyTorch或sklearn)
        model_path: 保存路径
    """
    try:
        # 确保目录存在
        ensure_directory(os.path.dirname(model_path))
        
        # 根据文件扩展名判断保存方式
        if model_path.endswith('.pkl'):
            # 使用pickle保存传统机器学习模型
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logging.info(f"传统机器学习模型已保存至: {model_path}")
        else:
            # 使用PyTorch保存深度学习模型
            torch.save(model.state_dict(), model_path)
            logging.info(f"PyTorch模型已保存至: {model_path}")
    
    except Exception as e:
        logging.error(f"保存模型错误: {str(e)}")

def load_csv(file_path):
    """
    加载CSV文件
    
    参数:
        file_path: CSV文件路径
    
    返回:
        df: Pandas DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"成功加载CSV文件: {file_path}, 行数: {len(df)}")
        return df
    
    except Exception as e:
        logging.error(f"加载CSV文件错误: {str(e)}")
        return pd.DataFrame()

def evaluate_model(y_true, y_pred, labels=None):
    """
    评估模型性能并计算指标
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        labels: 标签列表（可选）
    
    返回:
        metrics: 包含各种评估指标的字典
    """
    try:
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 计算指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # 生成分类报告
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # 构建结果字典
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'report': report
        }
        
        logging.info(f"模型评估指标: 准确率={accuracy:.4f}, 精确率={precision:.4f}, 召回率={recall:.4f}, F1分数={f1:.4f}")
        
        return metrics
    
    except Exception as e:
        logging.error(f"模型评估错误: {str(e)}")
        return {}

def create_timestamp():
    """
    创建时间戳字符串
    
    返回:
        timestamp: 时间戳字符串 (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def sampling_data(df, n_samples=None, random_state=42):
    """
    对数据进行采样
    
    参数:
        df: Pandas DataFrame
        n_samples: 采样数量 (若为None则返回原始数据)
        random_state: 随机种子
    
    返回:
        sampled_df: 采样后的DataFrame
    """
    try:
        if n_samples is None or n_samples >= len(df):
            return df
        
        return df.sample(n=n_samples, random_state=random_state)
    
    except Exception as e:
        logging.error(f"数据采样错误: {str(e)}")
        return df
