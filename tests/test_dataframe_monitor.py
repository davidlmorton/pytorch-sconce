from sconce.monitors import DataframeMonitor
import unittest
import tempfile


class TestDataframeMonitor(unittest.TestCase):
    def test_save_and_load(self):
        metadata = {'foo': [{'bar': 'baz'}]}
        monitor = DataframeMonitor(metadata=metadata)
        monitor.step(data={'a': 1, 'b': 10})
        monitor.step(data={'a': 3, 'b': 13})

        with tempfile.NamedTemporaryFile() as ofile:
            temp_filename = ofile.name

        monitor.save(filename=temp_filename, key='blah')

        monitor2 = DataframeMonitor.from_file(temp_filename, key='blah')

        self.assertEqual(monitor2.metadata, metadata)
        self.assertEqual(monitor2.df.a[0], 1)
        self.assertEqual(monitor2.df.a[1], 3)
        self.assertEqual(monitor2.df.b[0], 10)
        self.assertEqual(monitor2.df.b[1], 13)
