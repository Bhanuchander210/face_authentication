import os
import numpy as np
import shutil
import logging

logging.basicConfig(filename='face_auth_logs.out', format='%(asctime)-15s : %(filename)s:%(lineno)s : %(funcName)s() : %(message)s', filemode='a',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)


default_threshold = 0.6


def get_logging():
    return logging


def clear_dir_if_exists(input_dir):
    if os.path.exists(input_dir):
        shutil.rmtree(input_dir)


def get_all_dir(input_dir):
    return [x for x in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, x))]


def get_all_files(input_dir):
    return [x for x in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, x))]


class EmbedsUtils:
    def __init__(self, path):
        self.path = path
        self.embeds_file = os.path.join(path, 'emb_array.npy')
        self.labels_file = os.path.join(path, 'label_array.npy')
        self.create_if_not_exists()

    def clear_dir_if_exists(self):
        clear_dir_if_exists(self.path)

    def create_if_not_exists(self):
        if not os.path.isdir(self.path):
            os.makedirs(self.path)

    def get_embeds_data(self):
        return np.load(self.embeds_file)

    def get_labels_data(self):
        return np.load(self.labels_file)


def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


def minimum_distance_classifier(emb_util, face_encoding_to_check):
    embed_data = emb_util.get_embeds_data()
    label_data = emb_util.get_labels_data()
    result = {}
    for x in list(set(label_data)):
        class_indices = np.nonzero(np.array(label_data) == x)[0]
        x_embed_data = np.take(embed_data, list(class_indices), 0)
        distances = face_distance(x_embed_data, face_encoding_to_check)
        avg = sum(distances)/len(distances)
        logging.info("[Test] : Average distance value : {} for the class : {}".format(str(avg), x))
        result[x] = avg
    result_class = min(result, key=result.get)
    logging.info("[Result] : Minimum distance observed : {} for the class : {}".format(str(result[result_class]),
                                                                                       result_class))
    return [result[result_class], result_class]
