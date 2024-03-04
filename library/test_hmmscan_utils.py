import unittest
import hmmscan_utils as hu

class TestParseHmmscanResults(unittest.TestCase):
    def setUp(self):
    
        self.file_path = "library/test_data/test_hmmscan.txt"
        self.e_value_threshold = 0.01
        self.target_prob = 0.8
        self.length_thresh = 10
        self.result = hu.parse_hmmscan_results(self.file_path, self.e_value_threshold, self.target_prob, self.length_thresh)
    
    def test_dictionary_full(self):
        # Check that the dictionary is not empty
        self.assertNotEqual(len(self.result), 0)
    
    def test_dictionary_keys(self):
        # Check that each key is a string
        for key in self.result.keys():
            self.assertIsInstance(key, str)
    
    def test_dictionary_length(self):
        # test that there are two keys for this example
        self.assertEqual(len(self.result), 2)

    def test_dictionary_values(self):
        '''
        Dictionary should be in the following format:
        {Key (str): {'length': length (int), 'hit_domains': [[start (int), end (int), prob (float), fam (str), MID (str)], ...]} (list)}
        '''

        for key, value in self.result.items():
            self.assertIsInstance(value['length'], int)
            self.assertIsInstance(value['hit_domains'], list)
            
            for hit_domain in value['hit_domains']:
                self.assertEqual(len(hit_domain), 5)
                self.assertIsInstance(hit_domain[0], int)
                self.assertIsInstance(hit_domain[1], int)
                self.assertIsInstance(hit_domain[2], float)
                self.assertIsInstance(hit_domain[3], str)
                self.assertIsInstance(hit_domain[4], str)
    
    def test_translate_to_MID(self):
        # Check that the sequence is translated to M/I/D correctly and smoothed per MSS
        seq = 'MMMMMMMMIIIIMIIIMIIIMMMMM'
        result = hu.translate_to_MID(seq, self.target_prob, self.length_thresh)
        self.assertEqual(result, 'MMMMMMMMMMMMMMMMMMMMMMMMM')

if __name__ == '__main__':
    unittest.main()