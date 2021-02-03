import pathlib
import os
import pickle


def join(*paths):
    paths = [str(path) for path in paths]
    return str(pathlib.Path(*paths))


def mkdir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def exist(path):
    return os.path.exists(str(path))


def parent(path):
    path = pathlib.Path(path)
    return str(path.parent)


def is_file(path):
    if not exist(path):
        raise Exception(f"path ({path}) not exist")
    path = pathlib.Path(path)
    return path.is_file()


def file_name(path):
    path = pathlib.Path(path)
    return path.name


def readpickle(path, pickle_module=pickle):
    if not exist(path) or not is_file(path):
        raise Exception(f"path ({path}) not exist or not file")

    with open(path, "rb") as fid:
        data = pickle_module.load(fid)
    return data


def writepickle(path, data, pickle_module=pickle):
    if not exist(parent(path)):
        mkdir(parent(path))

    with open(path, "wb") as fid:
        pickle_module.dump(data, fid, protocol=4)
