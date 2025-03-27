from app import db
from datetime import datetime

class SMSMessage(db.Model):
    """短信消息模型"""
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    send_freq = db.Column(db.Float, default=0.0)  # 发送频率特征
    is_night = db.Column(db.Integer, default=0)   # 是否夜间发送特征 (0:否, 1:是)
    prediction = db.Column(db.String(10), nullable=True)  # 预测结果 (垃圾短信/正常短信)
    confidence = db.Column(db.Float, default=0.0)  # 预测置信度
    model_type = db.Column(db.String(20), nullable=True)  # 使用的模型类型
    timestamp = db.Column(db.DateTime, default=datetime.now)  # 记录时间
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=True)  # 关联的数据集ID
    
    def __repr__(self):
        return f"<SMS {self.id}: {self.text[:20]}... - {self.prediction}>"

class Dataset(db.Model):
    """数据集模型"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)  # 存储文件的路径
    description = db.Column(db.Text, nullable=True)
    total_records = db.Column(db.Integer, default=0)
    spam_count = db.Column(db.Integer, default=0)
    ham_count = db.Column(db.Integer, default=0)
    upload_time = db.Column(db.DateTime, default=datetime.now)
    last_used = db.Column(db.DateTime, nullable=True)
    
    # 建立与短信消息的一对多关系
    messages = db.relationship('SMSMessage', backref='dataset', lazy=True)
    
    def __repr__(self):
        return f"<Dataset {self.id}: {self.name} ({self.total_records} records)>"
