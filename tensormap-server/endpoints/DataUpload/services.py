import os
import pandas as pd
from endpoints.DataUpload.models import DataFile
from flask import request
from shared.constants import *
from shared.request.response import generic_response
from shared.services.config import get_configs
from shared.utils import delete_one_record, save_one_record
from werkzeug.utils import secure_filename

configs = get_configs()
upload_folder = configs['api']['upload']['folder']


def add_file_service():
    try:
        # Extract the file and save it in the ./data folder
        file = request.files['data']
        filename = secure_filename(file.filename.lower())
        file.save(os.path.join(upload_folder, filename))

        # Extract the file name and type and save details in the database

        # file_name has to be passed through secure_filename function to store the same name in the db
        file_name_db = secure_filename(file.filename.rsplit('.', 1)[0].lower())
        file_type_db = file.filename.rsplit('.', 1)[1].lower()
        data = DataFile(file_name=file_name_db, file_type=file_type_db)
        save_one_record(record=data)
        return generic_response(status_code=201, success=True, message='File saved successfully')
    except Exception as e:
        return generic_response(status_code=500, success=False, message='An error occurred while adding the file: {}'.format(str(e)))


def get_all_files_service():
    try:
        data = []
        files = DataFile.query.all()
        for file in files:
            df = pd.read_csv(upload_folder + '/' + file.file_name + '.' + file.file_type)
            fields = list(df.columns)
            data.append({FILE_NAME: file.file_name, FILE_TYPE: file.file_type, FILE_ID: file.id, FILE_FIELDS: fields})
        return generic_response(status_code=200, success=True, message='Saved files found successfully', data=data)
    except Exception as e:
        return generic_response(status_code=500, success=False, message='An error occurred while retrieving files: {}'.format(str(e)))


def delete_one_file_by_id_service(file_id):
    try:
        # Check if file exists in DB
        file = DataFile.query.get(file_id)
        if file:
            file_path = os.path.join(upload_folder, '{}.{}'.format(file.file_name, file.file_type))
            # Check if file exists in upload folder
            if os.path.isfile(file_path):
                os.remove(file_path)
                delete_one_record(record=file)
                return generic_response(status_code=200, success=True, message='File deleted successfully')
            else:
                return generic_response(status_code=400, success=False, message='File not found')
        else:
            return generic_response(status_code=400, success=False, message='File not found in the database')
    except Exception as e:
        return generic_response(status_code=500, success=False, message='An error occurred while deleting the file: {}'.format(str(e)))
