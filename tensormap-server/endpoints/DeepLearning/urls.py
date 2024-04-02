from endpoints.DeepLearning.views import GetCode, GetModelList, RunCode, ValidateModel
from shared.constants import *
from shared.services.config import get_configs

def model_urls(api):
    try:
        configs = get_configs()
        base = configs['api']['base']
        learn_uri = configs['api']['model']['uri']

        api.add_resource(ValidateModel, base + learn_uri + URL_VALIDATE)
        api.add_resource(GetCode, base + learn_uri + URL_CODE)
        api.add_resource(GetModelList, base + learn_uri + URL_GET_MODEL_LIST)
        api.add_resource(RunCode, base + learn_uri + URL_RUN)
    except Exception as e:
        print(f"Error setting up model URLs: {str(e)}")
