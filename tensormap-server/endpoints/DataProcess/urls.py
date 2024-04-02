from endpoints.DataProcess.views import ProcessAddNGet, ProcessIDOperations, GetDataMetrics
from shared.services.config import get_configs
from flask_restful import Api
from werkzeug.exceptions import HTTPException

configs = get_configs()

def process_urls(api: Api):
    try:
        base = configs['api']['base']
        data_uri = configs['api']['process']['uri']

        api.add_resource(ProcessAddNGet, base + data_uri + '/target')
        api.add_resource(ProcessIDOperations, base + data_uri + '/target/<int:file_id>')
        api.add_resource(GetDataMetrics, base + data_uri + '/data_metrics/<int:file_id>')
    except Exception as e:
        # Handle the exception appropriately
        raise HTTPException(description=f"An error occurred while adding resources: {e}")
