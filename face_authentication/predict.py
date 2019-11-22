import argparse
from face_authentication import utils
import face_recognition


def main(test_image, embeds_dir):
    emb_util = utils.EmbedsUtils(embeds_dir)
    loaded_image = face_recognition.load_image_file(test_image)
    test_data = face_recognition.face_encodings(loaded_image)[0]
    distance, index = utils.compare_faces(emb_util.get_embeds_data(), test_data)
    if distance <= utils.default_threshold:
        print("Face authorized as : {}".format(str(emb_util.get_labels_data()[index])))
    else:
        print("Unauthorized person.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--test-image', type=str, action='store', dest='test_image',
                        help='Input path of data to train on')
    parser.add_argument('--embeds-dir', type=str, action='store', dest='embeds_dir',
                        help='Path to output of the embeds')
    args = parser.parse_args()
    main(test_image=args.test_image, embeds_dir=args.embeds_dir)
