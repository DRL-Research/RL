import os

def get_latest_model(directory):
    '''

    Get last model from all directory. file must end with .zip
    :param directory:
    :return: last model
    '''
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".zip"):
                files.append(os.path.join(root, filename))

    if not files:
        return None

    latest_file = max(files, key=os.path.getctime)
    print('Latest model:', latest_file)
    return latest_file

def get_model_from_specific_directory(directory):
    '''
    return the last model from specific directory

    :param directory:
    :return: last model
    '''
    relevant_directory=os.chdir(directory)
    for file in os.listdir(relevant_directory):
        if file.endswith(".zip"):
            return file
