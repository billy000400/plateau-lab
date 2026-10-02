"""Shared local/hosted word boundaries, look-ahead, EOS and token-limit behavior."""
import unittest
from plateau.core.generation import word_prediction


class Tokenizer:
    eos_token_id = 99
    pieces = {0: 'It was big', 1: 'ger', 2: ' than', 3: ' my', 4: ' house', 5: '.', 6: ' Next'}

    def decode(self, tokens, **kwargs):
        return ''.join(self.pieces.get(token, '…') for token in tokens)


class Generation(unittest.TestCase):
    def test_three_words_with_subword_and_lookahead(self):
        value=word_prediction(Tokenizer(),[0],[1,2,3,4,5,6],[.5]*6)
        self.assertEqual(value['words'],['than','my','house'])
        self.assertEqual(value['continuation'],'ger than my house')
        self.assertTrue(value['complete'])
        self.assertEqual(len(value['tokens']),6)

    def test_eos_discards_forced_later_tokens(self):
        value=word_prediction(Tokenizer(),[0],[2,99,3,4,6],[.5]*5)
        self.assertEqual(value['words'],['than'])
        self.assertTrue(value['complete'])
        self.assertEqual(len(value['tokens']),1)

    def test_limit_is_not_a_complete_word_boundary(self):
        value=word_prediction(Tokenizer(),[0],[2,3,4],[.5]*3)
        self.assertEqual(value['word_count'],3)
        self.assertFalse(value['complete'])
        self.assertEqual(word_prediction(Tokenizer(),[0],[99,2],[.5,.5])['word_count'],0)


if __name__=='__main__':
    unittest.main()
