import unittest
import preprocess.data as data


class TestData(unittest.TestCase):
    def test_save_succeed(self):
        data.save('/home/hack/PycharmProjects/vgg/data/bim/1008',
                  '/home/hack/PycharmProjects/vgg/data/bim/1008/data',
                  '/home/hack/PycharmProjects/vgg/data/bim/1008/label.pickle')
