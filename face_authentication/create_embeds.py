import face_recognition
import os
import argparse
import numpy as np
from face_authentication import utils


def do_save_embeds(input_dir, embeds_dir):
    master_embeds_array = None
    master_labels_array = list()
    for class_name in utils.get_all_dir(input_dir):
        for image_file in utils.get_all_files(os.path.join(input_dir, class_name)):
            loaded_image = face_recognition.load_image_file(os.path.join(input_dir, class_name, image_file))
            embed = face_recognition.face_encodings(loaded_image)[:1]
            master_embeds_array = embed if master_embeds_array is None else np.concatenate([master_embeds_array, embed])
            master_labels_array.append(class_name)
    embed_util = utils.EmbedsUtils(embeds_dir)
    utils.get_logging().info("Embeds created : {}, for classes : {}.".format(str(len(master_embeds_array)),
                                                                             str(",".join(np.unique(master_labels_array)))))
    np.save(embed_util.embeds_file, master_embeds_array)
    np.save(embed_util.labels_file, master_labels_array)


def main(input_dir, embeds_dir):
    utils.clear_dir_if_exists(embeds_dir)
    do_save_embeds(input_dir, embeds_dir)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--input-dir', type=str, action='store', dest='input_dir',
                        help='Input path of data to train on')
    parser.add_argument('--embeds-dir', type=str, action='store', dest='embeds_dir',
                        help='Path to output of the embeds')
    args = parser.parse_args()
    main(input_dir=args.input_dir, embeds_dir=args.embeds_dir)
