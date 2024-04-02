from endpoints.DataProcess.services import (
    add_target_service,
    delete_one_target_by_id_service,
    get_all_targets_service,
    get_one_target_by_id_service,
    get_data_metrics
)
from endpoints.DataProcess.validators import target_add_post_validator
from flask_restful import Resource


class ProcessAddNGet(Resource):
    def post(self):
        try:
            validator = target_add_post_validator()
            args = validator.parse_args()
            return add_target_service(incoming=args)
        except Exception as e:
            # Handle the exception appropriately
            return {'message': 'An error occurred while processing the request.', 'error': str(e)}, 500

    def get(self):
        try:
            return get_all_targets_service()
        except Exception as e:
            # Handle the exception appropriately
            return {'message': 'An error occurred while processing the request.', 'error': str(e)}, 500


class ProcessIDOperations(Resource):
    def delete(self, file_id):
        try:
            return delete_one_target_by_id_service(file_id=file_id)
        except Exception as e:
            # Handle the exception appropriately
            return {'message': 'An error occurred while processing the request.', 'error': str(e)}, 500

    def get(self, file_id):
        try:
            return get_one_target_by_id_service(file_id=file_id)
        except Exception as e:
            # Handle the exception appropriately
            return {'message': 'An error occurred while processing the request.', 'error': str(e)}, 500


class GetDataMetrics(Resource):
    def get(self, file_id):
        try:
            return get_data_metrics(file_id=file_id)
        except Exception as e:
            # Handle the exception appropriately
            return {'message': 'An error occurred while processing the request.', 'error': str(e)}, 500
