from __future__ import absolute_import

import os
import errno
from argparse import ArgumentParser

def opts_parser():

    usage = "LFNet Test"
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '-D', '--path', type=str, nargs='?', dest='path', metavar='PATH',
        help='Loading 4D training and validation LF from this path: (default: %(default)s)')
    parser.add_argument(
        '--scenes', type=str, nargs='+', metavar='SCENES', help='Namelist of LF scenes')
    parser.add_argument(
        '--model_path', type=str, nargs='?', metavar='MODEL_PATH',
        help='Loading pre-trained model file from this path: (default: %(default)s)')
    parser.add_argument(
        '--save_path', type=str, nargs='?', metavar='SAVE_PATH',
        help='Save Upsampled LF to this path: (default: %(default)s)')
    parser.add_argument(
        '-F', '--factor', type=int, default=4, metavar='FACTOR',
        choices=[2,3,4], help='Angular Upsampling factor: (default: %(default)s)')
    parser.add_argument(
        '-T', '--train_length', type=int, default=7, metavar='TRAIN_LENGTH',
        choices=[7,9], help='Training data length: (default: %(default)s)')
    parser.add_argument(
        '-C', '--crop_length', type=int, default=7, metavar='CROP_LENGTH',
        help='Crop Length from Initial LF: (default: %(default)s)')
    parser.add_argument(
        '-S', '--save_results', dest='save_results', action='store_true',
        help='Save Results or Not')

    return parser



def getSceneNameFromPath(path,ext):
    sceneNamelist = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(ext):
                sceneName = os.path.splitext(name)[0]
                sceneNamelist.append(sceneName)

    sceneNamelist.sort()

def del_files(path,ext):
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(ext):
                os.remove(os.path.join(root, name))

def mkdir_p(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def isfile(fname):
    return os.path.isfile(fname) 

def isdir(dirname):
    return os.path.isdir(dirname)

def join(path, *paths):
    return os.path.join(path, *paths)
