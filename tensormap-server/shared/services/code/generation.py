from endpoints.DataUpload.models import DataFile
from jinja2 import Environment, FileSystemLoader
from shared.constants import *
from shared.services.config import get_configs
from endpoints.DeepLearning.models import ModelBasic
from endpoints.DataProcess.models import ImageProperties


def code_generation(code_params):
    """
    This service is used to generate code

    :param code_params: parameters used to generate the code
    :return: success of the operation
    """
    dataset = code_params[DATASET]
    model = code_params[LEARNING_MODEL]

    data = {
        DATASET: {
            FILE_NAME: helper_generate_file_location(file_id=dataset[FILE_ID]),
            FILE_TARGET: dataset[FILE_TARGET],
            MODEL_TRAINING_SPLIT: dataset[MODEL_TRAINING_SPLIT],
        },
        LEARNING_MODEL: {
            MODEL_JSON_FILE: helper_generate_json_model_file_location(model_name=model[MODEL_NAME]),
            MODEL_OPTIMIZER: model[MODEL_OPTIMIZER],
            MODEL_METRIC: model[MODEL_METRIC],
            MODEL_EPOCHS: model[MODEL_EPOCHS],
        },
    }

    template_loader = FileSystemLoader(searchpath=TEMPLATE_ROOT)
    template_env = Environment(loader=template_loader)
    template = template_env.get_template(helper_map_correct_code_template(problem_type_id=code_params[PROBLEM_TYPE]))

    output = template.render(data=data)
    generated_code_file = open(CODE_GENERATION_LOCATION + model[MODEL_NAME] + CODE_GENERATION_TYPE, 'w+')
    generated_code_file.write(output + '\n')
    generated_code_file.close()
    return True

def generate_code(model_name):
    """
    This service is used to generate code

    :param code_params: parameters used to generate the code
    :return: success of the operation
    """
    model_configs = ModelBasic.query.filter_by(model_name=model_name).first()
    file_id = getattr(model_configs, FILE_ID)
    file = DataFile.query.filter_by(id=file_id).first()
    

    data = {
        DATASET: {
            FILE_NAME: helper_generate_file_location(file_id=getattr(model_configs, FILE_ID)),
            FILE_TARGET: getattr(model_configs,FILE_TARGET),
            MODEL_TRAINING_SPLIT: getattr(model_configs,MODEL_TRAINING_SPLIT),
        },
        LEARNING_MODEL: {
            MODEL_JSON_FILE: helper_generate_json_model_file_location(model_name=model_name),
            MODEL_OPTIMIZER: getattr(model_configs,MODEL_OPTIMIZER),
            MODEL_METRIC: getattr(model_configs,MODEL_METRIC),
            MODEL_EPOCHS: getattr(model_configs,MODEL_EPOCHS),
        },
    }
    if file.file_type == 'zip':
        image_prop = ImageProperties.query.filter_by(id=file_id).first()
        data[DATASET].update({
            IMG_SIZE: image_prop.image_size,
            BATCH_SIZE: image_prop.batch_size,
            COLOR_MODE: image_prop.color_mode,
            LABEL_MODE: image_prop.label_mode,
        })
        
    print(model_configs.model_type)
    template_loader = FileSystemLoader(searchpath=TEMPLATE_ROOT)
    template_env = Environment(loader=template_loader)
    template = template_env.get_template(helper_map_correct_code_template(problem_type_id=model_configs.model_type))

    output = template.render(data=data)
    return output



def helper_map_correct_code_template(problem_type_id):
    options = {1: CODE_TEMPLATE_FOLDER + 'multi-class-all-float-classification-csv.py', 
                2: CODE_TEMPLATE_FOLDER + 'linear-regression-all-float.py',
                3: CODE_TEMPLATE_FOLDER + 'simple-image-classification.py'}
    return options[problem_type_id]


def helper_generate_file_location(file_id):
    configs = get_configs()
    file = DataFile.query.filter_by(id=file_id).first()
    if file.file_type == 'zip':
        return configs['api']['upload']['folder'] + '/' + file.file_name
    return configs['api']['upload']['folder'] + '/' + file.file_name + '.' + file.file_type

def helper_get_image_properties(file_id):
    data = ImageProperties.query.filter_by(id=file_id).first()
    return data.image_size, data.batch_size, data.color_mode, data.label_mode

def helper_generate_json_model_file_location(model_name):
    return MODEL_GENERATION_LOCATION + model_name + MODEL_GENERATION_TYPE
