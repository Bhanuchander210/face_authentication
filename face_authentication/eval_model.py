import argparse
from face_authentication import utils
import face_recognition
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
            actual_count = actual_count + 1
            loaded_image = face_recognition.load_image_file(os.path.join(input_dir, class_name, image_file))
            test_data = face_recognition.face_encodings(loaded_image)[0]
            distance, index = utils.compare_faces(emb_util.get_embeds_data(), test_data)
            if distance <= utils.default_threshold:
                count = count + 1
                if class_name == emb_util.get_labels_data()[index]:
                    cpc = cpc + 1
                result[class_name].append(distance)
            logging.info("For Class : {}, Count : {}, Distance : {}".format(class_name, count, distance))
        accuracy = cpc / actual_count
        false_prediction = (count - cpc) / actual_count
        avg_distance = sum(result[class_name])/count
        max_distance = max(result[class_name])
        min_distance = min(result[class_name])
        result[class_name] = avg_distance
        print("class: {}, accuracy: {:.3f}, distances: {:.3f}-{:.3f}-{:.3f}, total: {}".format(class_name, accuracy,
                                                                               min_distance, avg_distance,
                                                                               max_distance, actual_count))
        if false_prediction:
            print ("Warning. False prediction found in class {} : {}".format(class_name, str(false_prediction)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--embeds-dir', type=str, action='store', dest='embeds_dir',
                        help='Path to output of the embeds')
    parser.add_argument('--test-dir', type=str, action='store', dest='test_dir',
                        help='Dir path to the test images.')
    args = parser.parse_args()
    main(embeds_dir=args.embeds_dir, input_dir=args.test_dir)
