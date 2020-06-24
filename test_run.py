import unittest
import os

from run import Evaluator


class TestRun(unittest.TestCase):
    class DummyConfig():
        image_dir = 'testdata/images'
        template_dir = 'testdata/templates'
        output_dir = 'testdata/output'
        use_cuda = False
        use_cython = False

    def setUp(self):
        config = self.DummyConfig()
        self.instance = Evaluator(config)

    def test_is_image(self):
        self.assertEqual(self.instance.is_img('aaa/bbb.png'), True)
        self.assertEqual(self.instance.is_img('aaa/bbb.jpg'), True)
        self.assertEqual(self.instance.is_img('aaa/bbb.txt'), False)

    def test_get_matched_templates(self):
        score_map = {
            'aaa/bbb.png': ([[],[]], [1.0, 2.0]),
            'aaa/ccc.png': ([[],[]], [3.0, 1.0]),
            'aaa/ddd.png': ([[],[]], [1.0, 1.0]),
        }
        result = self.instance.get_matched_templates(score_map, 2)

        self.assertEqual(result[0][0], 'aaa/ccc.png')
        self.assertEqual(result[0][2], 3.0)
        self.assertEqual(result[1][0], 'aaa/bbb.png')
        self.assertEqual(result[1][2], 2.0)

    def test_output_result(self):
        data = {
            'aaa/label0@aeara.png': ['aaa/label0.png', 'aaa/label1.png'],
            'aaa/label2@label3@aegq.png': ['aaa/label2.png', 'aaa/label3.png'],
        }
        accuracy, precision, recall = self.instance.output_result(data, False)
        self.assertEqual(int(accuracy), 75)
        self.assertEqual(int(precision), 75)
        self.assertEqual(int(recall), 100)

        data = {
            'aaa/none@wefwe.png': ['none.png'],
            'aaa/none@ewwgb.png': ['aaa/label0.png', 'aaa/label1.png'],
        }
        accuracy, precision, recall = self.instance.output_result(data, False)
        self.assertEqual(int(accuracy), 50)
        self.assertEqual(int(precision), 50)
        self.assertEqual(int(recall), 50)
