from endpoints.DataUpload.services import (
    add_file_service,
    delete_one_file_by_id_service,
    get_all_files_service,
)
from endpoints.DataUpload.validators import data_upload_post_validator
from flask_restful import Resource


class UploadDataFile(Resource):
    def post(self):
        validator = data_upload_post_validator()
        args = validator.parse_args()
        return add_file_service()

    def get(self):
        return get_all_files_service()


class UploadFileIDOperations(Resource):
    def delete(self, file_id):
        print(file_id)
        return delete_one_file_by_id_service(file_id=file_id)
