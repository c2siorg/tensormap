import os

import pandas as pd
import zipfile
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from endpoints.DataUpload.models import DataFile
import tensorflow as tf
from flask import request
from shared.constants import *
from shared.request.response import generic_response
from shared.services.config import get_configs
from shared.utils import delete_one_record, save_one_record
from werkzeug.utils import secure_filename

configs = get_configs()
upload_folder = configs['api']['upload']['folder']

def load_dataset(file_path, filename , timeout = 10):
    try: 
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                unzip_folder = os.path.join(upload_folder, filename.rsplit('.', 1)[0])
                zip_ref.extractall(unzip_folder)
        with tf.device('/CPU:0'):
            tf.keras.preprocessing.image_dataset_from_directory(unzip_folder)
            return True
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

def add_file_service():
    # Extract the file and save it in the ./data folder
    file = request.files['data']
    filename = secure_filename(file.filename.lower())
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)
    # Extract the file name and type and save details in the database

    if filename.endswith('.zip'):
        try:
            # Attempt to load the dataset
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(load_dataset, file_path, filename)
                result = future.result(timeout=10)  # 10 seconds timeout
            if(result == False):
                raise Exception("Error loading dataset")
            unzip_folder = os.path.join(upload_folder, filename.rsplit('.', 1)[0])
            # If successful, save details in the database (omitted for brevity)
            file_name_db = secure_filename(file.filename.rsplit('.', 1)[0].lower())
            file_type_db = file.filename.rsplit('.', 1)[1].lower()
            print(file_name_db, file_type_db)
            data = DataFile(file_name=file_name_db, file_type=file_type_db)
            save_one_record(record=data)
            return generic_response(status_code=201, success=True, message='File processed successfully')
        except Exception as e:
            # If an error occurs, delete the directory and the zip file
            unzip_folder = os.path.join(upload_folder, filename.rsplit('.', 1)[0])
            if os.path.exists(unzip_folder):
                shutil.rmtree(unzip_folder)
            os.remove(file_path)
            return generic_response(status_code=500, success=False, message='An error occurred while processing the file')
    # file_name has to be passed through secure_filename function to store the same name in the db
    file_name_db = secure_filename(file.filename.rsplit('.', 1)[0].lower())
    file_type_db = file.filename.rsplit('.', 1)[1].lower()
    print(file_name_db)
    data = DataFile(file_name=file_name_db, file_type=file_type_db)
    save_one_record(record=data)
    return generic_response(status_code=201, success=True, message='File saved successfully')


def get_all_files_service():
    try:
        data = []
        files = DataFile.query.all()
        count = 1
        for file in files:
            if(file.file_type == 'zip' and count == 3):
                print(file.file_name)
                count += 1
                # data.append({FILE_NAME: file.file_name, FILE_TYPE: file.file_type, FILE_ID: file.id, FILE_FIELDS: []})
            elif(file.file_type == 'zip'):
                count += 1
                continue
            else:
                df = pd.read_csv(upload_folder + '/' + file.file_name + '.' + file.file_type)
                fields = list(df.columns)
                data.append({FILE_NAME: file.file_name, FILE_TYPE: file.file_type, FILE_ID: file.id, FILE_FIELDS: fields})
        return generic_response(status_code=200, success=True, message='Saved files found successfully', data=data)
    except Exception as e:
        # Log the error message and return a generic response
        print(f"An error occurred: {str(e)}")
        return generic_response(status_code=500, success=False, message='An error occurred while fetching the files')


def delete_one_file_by_id_service(file_id):
    try:
        # Check file exists in DB and check the file in ./data directory if exist, file deleted
        if DataFile.query.filter_by(id=file_id).count() > 0:
            file = DataFile.query.filter_by(id=file_id).first()
            if os.path.isfile(upload_folder + '/' + file.file_name + '.' + file.file_type):
                os.remove(upload_folder + '/' + file.file_name + '.' + file.file_type)
                delete_one_record(record=file)
                return generic_response(status_code=200, success=True, message='Files deleted successfully')
            else:
                return generic_response(status_code=400, success=False, message='File not found')
        else:
            return generic_response(status_code=400, success=False, message='File not in the DB')
    except Exception as e:
        # Log the error message and return a generic response
        print(f"An error occurred: {str(e)}")
        return generic_response(status_code=500, success=False, message='An error occurred while deleting the file')

