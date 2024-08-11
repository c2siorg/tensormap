from shared.utils import get_db_ref

db = get_db_ref()


class DataFile(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    file_name = db.Column(db.String(100), nullable=False)
    file_type = db.Column(db.String(10), nullable=False)
    model_basic = db.relationship('ModelBasic', uselist=False, cascade='all,delete', backref='file')
    target = db.relationship('DataProcess', uselist=False, cascade='all,delete', backref='file')
    created_on = db.Column(db.DateTime, server_default=db.func.now())
    updated_on = db.Column(db.DateTime, server_default=db.func.now(), server_onupdate=db.func.now())

class ImageProperties(db.Model):
    id = db.Column(db.Integer, db.ForeignKey('data_file.id'), primary_key=True)
    image_size = db.Column(db.Integer, nullable=False)
    batch_size = db.Column(db.Integer, nullable=False)
    color_mode = db.Column(db.String(10), nullable=False)
    label_mode = db.Column(db.String(15), nullable=False)

    __table_args__ = (
        db.CheckConstraint(color_mode.in_(['grayscale', 'rgb', 'rgba']), name='color_mode_check'),
        db.CheckConstraint(label_mode.in_(['int', 'categorical', 'binary']), name='label_mode_check'),
    )