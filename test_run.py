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
