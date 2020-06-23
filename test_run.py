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
            'aaa/template0@111.png': ['aaa/template0.png', 'aaa/template1.png'],
            'aaa/template2@template3@111.png': ['aaa/template2.png', 'aaa/template3.png'],
        }
        accuracy = self.instance.output_result(data, False)
        self.assertEqual(int(accuracy), 75)
