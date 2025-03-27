from app import db
from models import SMSMessage
from datetime import datetime

def add_sms(text, send_freq, is_night, prediction, confidence, model_type):
    """添加一条短信记录到数据库"""
    sms = SMSMessage(
        text=text,
        send_freq=send_freq,
        is_night=is_night,
        prediction=prediction,
        confidence=confidence,
        model_type=model_type,
        timestamp=datetime.now()
    )
    db.session.add(sms)
    db.session.commit()
    return sms

def add_sms_batch(sms_list):
    """批量添加短信记录到数据库"""
    for sms_data in sms_list:
        sms = SMSMessage(
            text=sms_data['text'],
            send_freq=sms_data['send_freq'],
            is_night=sms_data['is_night'],
            prediction=sms_data['prediction'],
            confidence=sms_data['confidence'],
            model_type=sms_data['model_type'],
            timestamp=datetime.now()
        )
        db.session.add(sms)
    db.session.commit()

def get_all_sms():
    """获取所有短信记录"""
    return SMSMessage.query.all()

def get_recent_sms(limit=100):
    """获取最近的n条短信记录"""
    return SMSMessage.query.order_by(SMSMessage.timestamp.desc()).limit(limit).all()

def get_spam_sms():
    """获取所有垃圾短信记录"""
    return SMSMessage.query.filter_by(prediction='垃圾短信').all()

def get_ham_sms():
    """获取所有正常短信记录"""
    return SMSMessage.query.filter_by(prediction='正常短信').all()

def get_sms_by_model(model_type):
    """获取特定模型的短信记录"""
    return SMSMessage.query.filter_by(model_type=model_type).all()

def delete_sms(sms_id):
    """删除一条短信记录"""
    sms = SMSMessage.query.get(sms_id)
    if sms:
        db.session.delete(sms)
        db.session.commit()
        return True
    return False

def delete_multiple_sms(sms_ids):
    """批量删除多条短信记录"""
    if not sms_ids:
        return False
    
    # 使用 in_ 查询批量删除
    deleted = SMSMessage.query.filter(SMSMessage.id.in_(sms_ids)).delete(synchronize_session=False)
    db.session.commit()
    return deleted > 0

def delete_all_sms():
    """删除所有短信记录"""
    SMSMessage.query.delete()
    db.session.commit()
    return True

def get_sms(sms_id):
    """根据ID获取一条短信记录"""
    return SMSMessage.query.get(sms_id)
