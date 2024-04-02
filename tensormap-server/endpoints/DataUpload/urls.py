from endpoints.DataUpload.views import UploadDataFile, UploadFileIDOperations
from shared.services.config import get_configs
from werkzeug.exceptions import BadRequest

configs = get_configs()

def data_urls(api):
    try:
        base = configs['api']['base']
        data_uri = configs['api']['upload']['uri']

        api.add_resource(UploadDataFile, base + data_uri + '/file')
        api.add_resource(UploadFileIDOperations, base + data_uri + '/file/<int:file_id>')
    except Exception as e:
        # Handle the exception appropriately
        raise BadRequest(description=f"An error occurred while setting up data URLs: {e}")
