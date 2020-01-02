import argparse
from face_authentication import utils, create_embeds
import os
logging = utils.get_logging()


def main(embeds_dir, input_dir):
    emb_util = utils.EmbedsUtils(embeds_dir)
    result = dict()
    for class_name in utils.get_all_dir(input_dir):
        result[class_name] = list()
        count = 0
        actual_count = 0
        cpc = 0
        for image_file in utils.get_all_files(os.path.join(input_dir, class_name)):
            face_encoding = create_embeds.get_embed_array(os.path.join(input_dir, class_name, image_file))
            if not list(face_encoding):
                logging.info("Skipped : {}".format(image_file))
                continue
            actual_count = actual_count + 1
            test_data = face_encoding[0]
            distance, predicted_class = utils.minimum_distance_classifier(emb_util, test_data)
            if distance <= utils.default_threshold:
                count = count + 1
                if class_name == predicted_class:
                    cpc = cpc + 1
                result[class_name].append(distance)
            logging.info("For Class : {}, Count : {}, Distance : {}".format(class_name, count, distance))
        accuracy = cpc / actual_count
        false_prediction = (count - cpc) / actual_count
        avg_distance = -1
        max_distance = -1
        min_distance = -1
        if count != 0:
            avg_distance = sum(result[class_name])/count
            max_distance = max(result[class_name])
            min_distance = min(result[class_name])
            result[class_name] = avg_distance
        print("class: {}, accuracy: {:.3f}, distances: {:.3f}-{:.3f}-{:.3f}, total: {}".format(class_name, accuracy,
                                                                                               min_distance,
                                                                                               avg_distance,
                                                                                               max_distance,
                                                                                               actual_count))
        if false_prediction:
            print ("Warning. False prediction found in class {} : {}".format(class_name, str(false_prediction)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--embeds-dir', type=str, action='store', dest='embeds_dir',
                        help='Path to output of the embeds', required=True)
    parser.add_argument('--test-dir', type=str, action='store', dest='test_dir', help='Dir path to the test images.',
                        required=True)
    args = parser.parse_args()
    main(embeds_dir=args.embeds_dir, input_dir=args.test_dir)
