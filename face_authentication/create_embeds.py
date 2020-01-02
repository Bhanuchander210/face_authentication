import face_recognition
import os
import argparse
import numpy as np
from face_authentication import utils

from keras.models import load_model
from keras.utils import CustomObjectScope
import tensorflow as tf
import cv2

# Switch this method to use the OpenFace Model

# with CustomObjectScope({'tf': tf}):
#     model = load_model('/tmp/nn4.small2.lrn_old.h5')
#
#
# def get_embed_array(image_path):
#     loaded_image = face_recognition.load_image_file(image_path)
#     loaded_image = cv2.resize(loaded_image, dsize=(96, 96), interpolation=cv2.INTER_CUBIC)
#     loaded_image = np.array([loaded_image])
#     return model.predict_on_batch(loaded_image)


def get_embed_array(image_path):
    loaded_image = face_recognition.load_image_file(image_path)
    return face_recognition.face_encodings(loaded_image)[:1]


def do_save_embeds(input_dir, embeds_dir):
    master_embeds_array = None
    master_labels_array = list()
    for class_name in utils.get_all_dir(input_dir):
        for image_file in utils.get_all_files(os.path.join(input_dir, class_name)):
            embed = get_embed_array(os.path.join(input_dir, class_name, image_file))
            if not list(embed):
                continue
            master_embeds_array = embed if master_embeds_array is None else np.concatenate([master_embeds_array, embed])
            master_labels_array.append(class_name)
    utils.get_logging().info("Embeds created size : {}".format(str(len(master_embeds_array))))
    embed_util = utils.EmbedsUtils(embeds_dir)
    np.save(embed_util.embeds_file, master_embeds_array)
    np.save(embed_util.labels_file, master_labels_array)


def main(input_dir, embeds_dir):
    utils.clear_dir_if_exists(embeds_dir)
    do_save_embeds(input_dir, embeds_dir)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--input-dir', type=str, action='store', dest='input_dir',
                        help='Input path of data to train on', required=True)
    parser.add_argument('--embeds-dir', type=str, action='store', dest='embeds_dir',
                        help='Path to output of the embeds', required=True)
    args = parser.parse_args()
    main(input_dir=args.input_dir, embeds_dir=args.embeds_dir)
