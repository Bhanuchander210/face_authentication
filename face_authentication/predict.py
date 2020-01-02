import argparse
from face_authentication import utils, create_embeds


def main(test_image, embeds_dir):
    emb_util = utils.EmbedsUtils(embeds_dir)
    test_data = create_embeds.get_embed_array(test_image)
    distance, predicted_class = utils.minimum_distance_classifier(emb_util, test_data)
    if distance <= utils.default_threshold:
        print("Face authorized as : {} with distance : {}".format(predicted_class, str(distance)))
    else:
        print("Near to {}:{}. Unauthorized person.".format(predicted_class, distance))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--test-image', type=str, action='store', dest='test_image',
                        help='Input path of data to train on', required=True)
    parser.add_argument('--embeds-dir', type=str, action='store', dest='embeds_dir',
                        help='Path to output of the embeds', required=True)
    args = parser.parse_args()
    main(test_image=args.test_image, embeds_dir=args.embeds_dir)
