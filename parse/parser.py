import base64
import json
import os
import logging


# logging.basicConfig(level=logging.INFO)


class Base64Parser:
    def __init__(self):
        """
        Json keys constant and base64 prefix.
        """
        self.IMG_URL = 'imgURL'
        self.EYE = 'eye'
        self.LOOK = 'look'
        self.X = 'x'
        self.Y = 'y'
        self.Z = 'z'
        self.START = len('data:image/png;base64,')

    def parse_json_list(self, json_list_str, pics_dir, concat="--"):
        json_list = json.loads(json_list_str)
        for data in json_list:
            img, new_filename = self.parse_json(data, concat=concat)
            with open(os.path.join(pics_dir, new_filename), 'wb') as file:
                file.write(img)

    def walk(self, input_path, output_path):
        """
        Scan the input path ,parse and store the pared image in output path.

        :param input_path: Path contains the json files.
        :param output_path: Output path to store the parsed BIM images.
        :return:
        """
        assert os.path.isdir(input_path), "a directory is need"
        for dirpath, dirnames, filenames in os.walk(input_path):
            for filename in filenames:
                if filename.endswith('.json'):
                    img, new_filename = self.parse(os.path.join(dirpath, filename))
                    with open(os.path.join(output_path, new_filename), 'wb') as file:
                        file.write(img)
                        logging.info("Write parsed file: {}".format(new_filename))

    def parse(self, filename, concat='--'):
        """
        Parse a json file into a image and new filename

        :param filename: filename of the json file to be parsed.
        :param concat: concatenation sign.
        :return: parsed image and a new filename.
        """
        with open(filename) as json_file:
            return self.parse_json(json_file, concat=concat)

    def parse_list(self, filename, output_path, concat='--'):
        with open(filename) as json_file:
            for data in json.load(json_file):
                img, new_filename = self.parse_json(data, concat=concat)
                with open(os.path.join(output_path, new_filename), 'wb') as file:
                    file.write(img)

    def parse_json(self, data, concat='--'):
        img_base64 = data[self.IMG_URL]

        eye_x = data[self.EYE][self.X]
        eye_y = data[self.EYE][self.Y]
        eye_z = data[self.EYE][self.Z]

        look_x = data[self.LOOK][self.X]
        look_y = data[self.LOOK][self.Y]
        look_z = data[self.LOOK][self.Z]

        position = int(eye_x), int(eye_y), int(eye_z)
        direction = int(look_x - eye_x), int(look_y - eye_y), int(look_z - eye_z)

        img_base64 = img_base64[self.START:]
        img = base64.b64decode(img_base64)
        new_filename = str(position) + concat + str(direction) + '.png'
        return img, new_filename


if __name__ == "__main__":
    parser = Base64Parser()
    # parser.walk('/home/hack/PycharmProjects/vgg/parse', '/home/hack/PycharmProjects/vgg/data/bim/1008')
    parser.parse_list("/home/hack/PycharmProjects/vgg/data/bim/1008/allImageData(1).txt",
                      "/home/hack/PycharmProjects/vgg/data/bim/1008")
