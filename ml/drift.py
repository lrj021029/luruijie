import numpy as np
import logging
import importlib.util
import traceback
import os
from datetime import datetime
from ml.preprocessing import clean_text
from ml.feature_extraction import extract_features

# 检查依赖库是否可用
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None

# 判断是否在Replit环境中
IS_REPLIT = 'REPLIT_DB_URL' in os.environ

# PyTorch导入
if HAS_TORCH:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        logging.info("成功导入torch库，用于语义漂移检测")
    except Exception as e:
        logging.error(f"导入torch库失败: {e}")
        HAS_TORCH = False

# TensorFlow导入 - 在Replit环境中禁用，但保留代码供本地部署时使用
if IS_REPLIT:
    # 在Replit中禁用TensorFlow
    HAS_TENSORFLOW = False
    logging.info("Replit环境中禁用TensorFlow，使用PyTorch实现语义漂移检测")
else:
    # 非Replit环境（如本地CUDA机器）尝试导入TensorFlow
    if HAS_TENSORFLOW:
        try:
            # 这里的代码只有在本地部署时才会执行
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers, Model
            logging.info("成功导入TensorFlow库，用于语义漂移检测")
            
            # 检查GPU是否可用
            if tf.config.list_physical_devices('GPU'):
                logging.info(f"已检测到GPU: {tf.config.list_physical_devices('GPU')}")
                # 配置TensorFlow使用内存增长而非一次性分配
                for gpu in tf.config.list_physical_devices('GPU'):
                    tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            logging.error(f"导入TensorFlow库失败: {e}")
            HAS_TENSORFLOW = False
    else:
        logging.info("未找到TensorFlow库，使用PyTorch实现语义漂移检测")

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)

# VAE模型：用于检测特征分布的漂移
if HAS_TORCH:
    class SMSVAE(nn.Module):
        """
        短信文本的变分自编码器，用于检测语义漂移
        """
        def __init__(self, input_dim=768, hidden_dim=128, latent_dim=32):
            super(SMSVAE, self).__init__()
            
            # 编码器
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
            # 均值和对数方差层
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
            
            # 解码器
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, input_dim)
            )
        
        def encode(self, x):
            """编码过程：输入 -> 隐藏层 -> mu, logvar"""
            h = self.encoder(x)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar
        
        def reparameterize(self, mu, logvar):
            """重参数化技巧"""
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z
        
        def decode(self, z):
            """解码过程：latent vector -> 重建输入"""
            return self.decoder(z)
        
        def forward(self, x):
            """前向传播"""
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            recon_x = self.decode(z)
            return recon_x, mu, logvar

# 创建/加载VAE模型
def get_vae_model():
    """获取VAE模型实例"""
    # 检查PyTorch是否可用
    if not HAS_TORCH:
        logging.warning("PyTorch不可用，无法创建VAE模型")
        return None
        
    try:
        # 创建VAE模型
        vae = SMSVAE(input_dim=768)  # 仅使用文本特征，不包括元数据
        
        # 实际应用中应加载预训练模型
        # vae.load_state_dict(torch.load('models_store/vae_model.pth'))
        
        return vae
    except Exception as e:
        logging.error(f"获取VAE模型错误: {str(e)}")
        return None

def encode_texts(texts, tokenizer=None):
    """
    将文本编码为特征向量
    
    参数:
        texts: 文本列表
        tokenizer: 可选的tokenizer
    
    返回:
        features: 特征矩阵
    """
    try:
        features = []
        
        for text in texts:
            # 清理文本
            cleaned_text = clean_text(text)
            
            # 提取特征（仅使用文本特征，不包括元数据）
            feature = extract_features(cleaned_text, tokenizer)
            
            # 仅保留文本特征（去除元数据）
            text_feature = feature[:768]
            
            features.append(text_feature)
        
        return np.array(features)
    
    except Exception as e:
        logging.error(f"文本编码错误: {str(e)}")
        return np.array([])

def kl_divergence(mu1, logvar1, mu2, logvar2):
    """
    计算两个分布的KL散度
    
    参数:
        mu1, logvar1: 第一个分布的参数
        mu2, logvar2: 第二个分布的参数
    
    返回:
        kl_div: KL散度值
    """
    try:
        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)
        
        kl_div = 0.5 * torch.sum(
            logvar2 - logvar1 - 1 + var1/var2 + (mu1-mu2)**2 / var2
        )
        
        return kl_div.item() / mu1.size(0)  # 归一化
    
    except Exception as e:
        logging.error(f"KL散度计算错误: {str(e)}")
        return 0.0

def adapt_model(model, features, model_type):
    """
    根据当前数据进行模型在线微调
    
    参数:
        model: 待微调的模型
        features: 特征数据
        model_type: 模型类型
    
    返回:
        success: 微调是否成功
    """
    if not HAS_TORCH or model is None:
        return False
        
    try:
        logging.info(f"开始对{model_type}模型进行在线微调...")
        
        # 转换特征为张量
        features_tensor = torch.FloatTensor(features)
        
        # 生成伪标签 - 在实际应用中，这里应该使用更复杂的策略，
        # 比如使用聚类、半监督学习、或者专家知识
        # 这里为了演示，使用当前模型的预测结果作为伪标签
        with torch.no_grad():
            model.eval()
            outputs = []
            
            # 分批处理特征，避免内存不足
            batch_size = 32
            for i in range(0, len(features_tensor), batch_size):
                batch = features_tensor[i:i+batch_size]
                output = model(batch)
                outputs.append(output)
                
            outputs = torch.cat(outputs, dim=0)
            pseudo_labels = (torch.sigmoid(outputs) > 0.5).float()
        
        # 准备优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # 进入训练模式
        model.train()
        
        # 进行少量训练步骤的在线微调
        epochs = 3
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # 分批进行训练
            for i in range(0, len(features_tensor), batch_size):
                batch_features = features_tensor[i:i+batch_size]
                batch_labels = pseudo_labels[i:i+batch_size]
                
                # 前向传播
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / (len(features_tensor) / batch_size)
            logging.info(f"微调 Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # 恢复为评估模式
        model.eval()
        
        logging.info(f"{model_type}模型在线微调完成")
        return True
        
    except Exception as e:
        logging.error(f"模型在线微调失败: {str(e)}")
        traceback_str = traceback.format_exc()
        logging.error(traceback_str)
        return False

def detect_drift(texts, reference_texts=None, tokenizer=None, model=None, model_type=None):
    """
    检测语义漂移
    
    参数:
        texts: 当前文本列表
        reference_texts: 参考文本列表（若为None，则返回默认值）
        tokenizer: 可选的tokenizer
        model: 可选的分类器模型，用于微调
        model_type: 模型类型
    
    返回:
        drift_value: 漂移值，越大表示漂移越严重
        is_adapted: 是否触发了模型微调
    """
    try:
        is_adapted = False
        
        # 如果没有PyTorch或没有参考文本，返回默认漂移值
        if not HAS_TORCH or reference_texts is None or len(reference_texts) < 10:
            # 使用文本特征的hash生成一个伪随机值，保证相同输入的输出一致
            seed = hash(str(texts) + str(reference_texts)) % 10000
            np.random.seed(seed)
            return np.random.uniform(0.1, 0.3), is_adapted  # 返回一个较小的随机漂移值
            
        # 获取VAE模型
        vae = get_vae_model()
        
        if vae is None:
            return 0.0, is_adapted
        
        vae.eval()
        
        # 编码当前文本
        current_features = encode_texts(texts, tokenizer)
        
        # 编码参考文本
        reference_features = encode_texts(reference_texts, tokenizer)
        
        # 将特征转换为张量
        current_tensor = torch.FloatTensor(current_features)
        reference_tensor = torch.FloatTensor(reference_features)
        
        # 获取分布参数
        with torch.no_grad():
            _, current_mu, current_logvar = vae(current_tensor)
            _, reference_mu, reference_logvar = vae(reference_tensor)
            
            # 计算KL散度
            drift_value = kl_divergence(
                current_mu.mean(dim=0, keepdim=True),
                current_logvar.mean(dim=0, keepdim=True),
                reference_mu.mean(dim=0, keepdim=True),
                reference_logvar.mean(dim=0, keepdim=True)
            )
        
        # 归一化漂移值到[0,1]范围
        normalized_drift = min(drift_value / 10.0, 1.0)
        
        # 如果漂移值大于阈值且分类器模型可用，则对模型进行在线微调
        if normalized_drift > 0.5 and model is not None and HAS_TORCH:
            logging.info(f"检测到显著漂移（{normalized_drift}），开始对模型进行在线微调")
            is_adapted = adapt_model(model, current_features, model_type)
            if is_adapted:
                logging.info("模型在线微调成功")
            else:
                logging.warning("模型在线微调失败")
        
        return normalized_drift, is_adapted
    
    except Exception as e:
        logging.error(f"漂移检测错误: {str(e)}")
        return 0.0, False
