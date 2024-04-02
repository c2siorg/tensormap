from endpoints.DeepLearning.services import (
    get_available_model_list,
    get_code_service,
    model_validate_service,
    run_code_service,
)
from endpoints.DeepLearning.validators import (
    get_code_post_validator,
    model_validate_post_validator,
)
from flask_restful import Resource

class ValidateModel(Resource):
    def post(self):
        try:
            validator = model_validate_post_validator()
            if validator:
                args = validator.parse_args()
                return model_validate_service(incoming=args)
            else:
                return {"message": "Error occurred while validating model"}, 500
        except Exception as e:
            return {"message": f"An error occurred: {str(e)}"}, 500

class GetCode(Resource):
    def post(self):
        try:
            validator = get_code_post_validator()
            if validator:
                args = validator.parse_args()
                return get_code_service(incoming=args)
            else:
                return {"message": "Error occurred while validating code"}, 500
        except Exception as e:
            return {"message": f"An error occurred: {str(e)}"}, 500

class RunCode(Resource):
    def post(self):
        try:
            validator = get_code_post_validator()
            if validator:
                args = validator.parse_args()
                return run_code_service(incoming=args)
            else:
                return {"message": "Error occurred while running code"}, 500
        except Exception as e:
            return {"message": f"An error occurred: {str(e)}"}, 500

class GetModelList(Resource):
    def get(self):
        try:
            return get_available_model_list()
        except Exception as e:
            return {"message": f"An error occurred: {str(e)}"}, 500
