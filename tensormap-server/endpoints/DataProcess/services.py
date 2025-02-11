import pandas as pd

from endpoints.DataProcess.models import DataProcess, ImageProperties, db
from endpoints.DataUpload.models import DataFile
from shared.constants import *
from shared.request.response import generic_response
from shared.utils import delete_one_record, save_one_record
from shared.services.config import get_configs
from werkzeug.utils import secure_filename

configs = get_configs()
upload_folder = configs['api']['upload']['folder']

def add_target_service(incoming):
    file_id = incoming[FILE_ID]
    target = incoming[FILE_TARGET_FIELD]
    try:
        if DataFile.query.filter_by(id=file_id).count() > 0:
            data_process = DataProcess(file_id=file_id, file=DataFile.query.filter_by(id=file_id).first(), target=target)
            save_one_record(record=data_process)
            return generic_response(status_code=201, success=True, message='Target field added successfully')
        else:
            return generic_response(status_code=400, success=False, message="File doesn't exist in DB")
    except Exception as e:
        return generic_response(status_code=500, success=False, message=f"Error storing record: {str(e)}")


def get_all_targets_service():
    process_files = DataProcess.query.all()
    data = []
    for file in process_files:
        data.append(
            {
                FILE_ID: file.file_id,
                FILE_NAME: file.file.file_name,
                FILE_TYPE: file.file.file_type,
                FILE_TARGET: file.target,
            }
        )
    return generic_response(
        status_code=200, success=True, message='Target fields of all files received successfully', data=data
    )


def delete_one_target_by_id_service(file_id):
    if DataFile.query.filter_by(id=file_id).count() > 0:
        if DataProcess.query.filter_by(file_id=file_id).count() > 0:
            target_record = DataProcess.query.filter_by(file_id=file_id).first()
            delete_one_record(record=target_record)
            return generic_response(status_code=200, success=True, message='Target field deleted successfully')
        else:
            return generic_response(status_code=400, success=False, message="Target field doesn't exist")
    else:
        return generic_response(status_code=400, success=False, message="File doesn't exist in DB")


def get_one_target_by_id_service(file_id):
    if DataFile.query.filter_by(id=file_id).count() > 0:
        if DataProcess.query.filter_by(file_id=file_id).count() > 0:
            target_record = DataProcess.query.filter_by(file_id=file_id).first()
            data = {
                FILE_NAME: target_record.file.file_name,
                FILE_TYPE: target_record.file.file_type,
                FILE_TARGET: target_record.target,
            }
            return generic_response(
                status_code=200, success=True, message='Target fields of all files received successfully', data=data
            )
        else:
            return generic_response(status_code=400, success=False, message="Target field doesn't exist")

    else:
        return generic_response(status_code=400, success=False, message="File doesn't exist in DB")

def get_data_metrics(file_id):
    configs = get_configs()
    file = DataFile.query.filter_by(id=file_id).first()
    if file:
        FILE_NAME = configs['api']['upload']['folder'] + '/' + file.file_name + '.' + file.file_type
        df = pd.read_csv(FILE_NAME)
        metrics = {}
        metrics['data_types']  = df.dtypes.apply(str).to_dict()
        metrics['correlation_matrix'] = df.corr().applymap(str).to_dict()
        metrics['metric'] = df.describe().applymap(str).to_dict()
        # cov_matrix_rounded = np.around(cov_matrix.values, 2).tolist()
        return generic_response(
                    status_code=200, success=True, message='Dataset metrics generated succesfully', data=metrics
                )
    else:
        return generic_response(status_code=400, success=False, message="File doesn't exist in DB")

def get_file_data(file_id):
    configs = get_configs()
    file = DataFile.query.filter_by(id=file_id).first()
    if file:
        FILE_NAME = configs['api']['upload']['folder'] + '/' + file.file_name + '.' + file.file_type
        df = pd.read_csv(FILE_NAME)
        data_json = df.to_json(orient='records')
        return generic_response(
                    status_code=200, success=True, message='Data sent succesfully', data= data_json
                )
    else:
        return generic_response(status_code=400, success=False, message="Unable to open file")

def get_file_name(file_id):
    configs = get_configs()
    file = DataFile.query.filter_by(id=file_id).first()
    if file:
        return configs['api']['upload']['folder'] + '/' + file.file_name + '.' + file.file_type
    else:
        return None

def preprocess_data(file_id, data):
    transformations = data['transformations']
    file = get_file_name(file_id=file_id)
    if file:
        try:
            df = pd.read_csv(file)
            for transformation in transformations:
                if transformation['transformation'] == 'One Hot Encoding':
                    df = pd.get_dummies(df, columns=[transformation['feature']])
                if transformation['transformation'] == 'Categorical to Numerical':
                    df[transformation['feature']] = pd.Categorical(df[transformation['feature']]).codes
                if transformation['transformation'] == 'Drop Column':
                    df = df.drop(columns=[transformation['feature']])
            file_name = file.split('/')[-1].split('.')[0] + '_preprocessed.csv'
            file_name_db = secure_filename(file_name.rsplit('.', 1)[0].lower())
            file_type_db = file_name.rsplit('.', 1)[1].lower()
            data = DataFile(file_name=file_name_db, file_type=file_type_db)
            save_one_record(record=data)
            df.to_csv(upload_folder + '/' + file_name, index=False)
            return generic_response(status_code=200, success=True, message='Preprocessed Dataset created with name: ' + file_name)
        except Exception as e:
            return generic_response(status_code=500, success=False, message=f"Error preprocessing data: {str(e)}")
            
    return generic_response(status_code=400, success=False, message="File doesn't exist in DB")

def update_image_properties(file_id, file_type, image_size, batch_size, color_mode, label_mode):
    try:
        # Query the ImageProperties table using the file_id
        image_properties = ImageProperties.query.filter_by(id=file_id).first()

        if image_properties:
            # Update the existing record with new parameters
            image_properties.image_size = image_size
            image_properties.batch_size = batch_size
            image_properties.color_mode = color_mode
            image_properties.label_mode = label_mode
            db.session.commit()
            return generic_response(status_code=200, success=True, message='Image properties updated successfully')
        else:
            # Insert a new record with the provided parameters
            new_image_properties = ImageProperties(
                id=file_id,
                image_size=image_size,
                batch_size=batch_size,
                color_mode=color_mode,
                label_mode=label_mode
            )
            db.session.add(new_image_properties)
            db.session.commit()
            return generic_response(status_code=201, success=True, message='Image properties added successfully')
    except Exception as e:
        # Log the error message and return a generic response
        print(f"An error occurred: {str(e)}")
        return generic_response(status_code=500, success=False, message='An error occurred while upserting the image properties')

