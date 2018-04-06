from sconce.monitors import DataframeMonitor
import unittest
import tempfile


class TestDataframeMonitor(unittest.TestCase):
    def test_save_and_load(self):
        metadata = {'foo': [{'bar': 'baz'}]}
        monitor = DataframeMonitor(metadata=metadata)
        monitor.write(data={'a': 1, 'b': 10}, step=0.1)
        monitor.write(data={'a': 3, 'b': 13}, step=0.2)

        with tempfile.NamedTemporaryFile() as ofile:
            temp_filename = ofile.name

        monitor.save(filename=temp_filename, key='blah')

        monitor2 = DataframeMonitor.from_file(temp_filename, key='blah')

        self.assertEqual(monitor2.metadata, metadata)
        self.assertEqual(monitor2.df.a[0.1], 1)
        self.assertEqual(monitor2.df.a[0.2], 3)
        self.assertEqual(monitor2.df.b[0.1], 10)
        self.assertEqual(monitor2.df.b[0.2], 13)
