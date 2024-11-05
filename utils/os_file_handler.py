import os


def get_latest_model(directory):
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".zip"):
                files.append(os.path.join(root, filename))

    if not files:
        return None

    latest_file = max(files, key=os.path.getctime)
    return latest_file