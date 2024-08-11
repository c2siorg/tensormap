from endpoints.DataProcess.services import (
    add_target_service,
    delete_one_target_by_id_service,
    get_all_targets_service,
    get_file_data,
    get_one_target_by_id_service,
    get_data_metrics,
    get_file_name, 
    preprocess_data,
    update_image_properties
)
from endpoints.DataProcess.validators import target_add_post_validator
from flask_restful import Resource
from flask import request
from shared.request.response import generic_response


class ProcessAddNGet(Resource):
    def post(self):
        validator = target_add_post_validator()
        args = validator.parse_args()
        return add_target_service(incoming=args)

    def get(self):
        return get_all_targets_service()


class ProcessIDOperations(Resource):
    def delete(self, file_id):
        return delete_one_target_by_id_service(file_id=file_id)

    def get(self, file_id):
        return get_one_target_by_id_service(file_id=file_id)
    
class GetDataMetrics(Resource):
    def get(self,file_id):
        return get_data_metrics(file_id=file_id)

class GetFileData(Resource):
    def get(self, file_id):
        return get_file_data(file_id=file_id)

class PreprocessData(Resource):
    def post(self, file_id):
        data = request.get_json()
        return preprocess_data(file_id=file_id, data=data)

class ImagePreprocessProps(Resource):
    def post(self, file_id):
        data = request.get_json()
        return update_image_properties(file_id=data["fileId"], file_type=data["fileType"], image_size=data["image_size"], batch_size=data["batch_size"], color_mode=data["color_mode"], label_mode=data["label_mode"])