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
    
    def __repr__(self):
        return f"<SMS {self.id}: {self.text[:20]}... - {self.prediction}>"
