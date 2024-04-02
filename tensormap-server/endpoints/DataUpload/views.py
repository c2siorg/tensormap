from endpoints.DataUpload.services import (
    add_file_service,
    delete_one_file_by_id_service,
    get_all_files_service,
)
from endpoints.DataUpload.validators import data_upload_post_validator
from flask_restful import Resource
from werkzeug.exceptions import BadRequest

class UploadDataFile(Resource):
    def post(self):
        try:
            validator = data_upload_post_validator()
            args = validator.parse_args()
            return add_file_service()
        except Exception as e:
            # Handle the exception appropriately
            return {'message': f'An error occurred: {str(e)}'}, 500

    def get(self):
        try:
            return get_all_files_service()
        except Exception as e:
            # Handle the exception appropriately
            return {'message': f'An error occurred: {str(e)}'}, 500


class UploadFileIDOperations(Resource):
    def delete(self, file_id):
        try:
            return delete_one_file_by_id_service(file_id=file_id)
        except Exception as e:
            # Handle the exception appropriately
            return {'message': f'An error occurred: {str(e)}'}, 500
