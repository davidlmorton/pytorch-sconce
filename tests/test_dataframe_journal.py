from sconce.journals import DataframeJournal
import unittest
import tempfile


class TestDataframeJournal(unittest.TestCase):
    def test_save_and_load(self):
        metadata = {'foo': [{'bar': 'baz'}]}
        j = DataframeJournal(metadata=metadata)
        j.record_step(data={'a': 1, 'b': 10})
        j.record_step(data={'a': 3, 'b': 13})

        with tempfile.NamedTemporaryFile() as ofile:
            temp_filename = ofile.name

        j.save(filename=temp_filename, key='blah')

        j2 = DataframeJournal.from_file(temp_filename, key='blah')

        self.assertEqual(j2.metadata, metadata)
        self.assertEqual(j2.df.a[0], 1)
        self.assertEqual(j2.df.a[1], 3)
        self.assertEqual(j2.df.b[0], 10)
        self.assertEqual(j2.df.b[1], 13)
