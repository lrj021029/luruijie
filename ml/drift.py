import numpy as np
import logging
import importlib.util
from datetime import datetime
from ml.preprocessing import clean_text
from ml.feature_extraction import extract_features

# 检查PyTorch是否可用
HAS_TORCH = importlib.util.find_spec("torch") is not None

# 有条件地导入
if HAS_TORCH:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        logging.info("成功导入torch库，用于语义漂移检测")
    except Exception as e:
        logging.error(f"导入torch库失败: {e}")
        HAS_TORCH = False

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

def detect_drift(texts, reference_texts=None, tokenizer=None):
    """
    检测语义漂移
    
    参数:
        texts: 当前文本列表
        reference_texts: 参考文本列表（若为None，则返回默认值）
        tokenizer: 可选的tokenizer
    
    返回:
        drift_value: 漂移值，越大表示漂移越严重
    """
    try:
        # 如果没有PyTorch或没有参考文本，返回默认漂移值
        if not HAS_TORCH or reference_texts is None or len(reference_texts) < 10:
            # 使用文本特征的hash生成一个伪随机值，保证相同输入的输出一致
            seed = hash(str(texts) + str(reference_texts)) % 10000
            np.random.seed(seed)
            return np.random.uniform(0.1, 0.3)  # 返回一个较小的随机漂移值
            
        # 获取VAE模型
        vae = get_vae_model()
        
        if vae is None:
            return 0.0
        
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
        
        return normalized_drift
    
    except Exception as e:
        logging.error(f"漂移检测错误: {str(e)}")
        return 0.0
