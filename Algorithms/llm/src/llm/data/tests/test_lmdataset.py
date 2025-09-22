import unittest
import torch

class TiktokenTokenizer:
    def __init__(self, encoding_name: str = "cl100k_base"):
        import tiktoken
        self.enc = tiktoken.get_encoding(encoding_name)
    def encode(self, txt: str):
        return self.enc.encode(txt)

from llm.data.lmdataset import LMDataset  # or inline the class

class TestLMDataset(unittest.TestCase):
    def setUp(self):
        self.text = "hello world, this is a test case"
        self.tokenizer = TiktokenTokenizer("cl100k_base")
        self.context_length = 5
        self.stride = 5

    def test_len_positive(self):
        ds = LMDataset(self.text, self.tokenizer, self.context_length, self.stride)
        self.assertGreater(len(ds), 0)  # dataset yields at least one window [web:19]

    def test_item_shapes_and_dtypes(self):
        ds = LMDataset(self.text, self.tokenizer, self.context_length, self.stride)
        x, y = ds[0]
        self.assertTrue(torch.is_tensor(x))  # tensors returned [web:19]
        self.assertTrue(torch.is_tensor(y))  # tensors returned [web:19]
        self.assertEqual(x.dtype, torch.long)  # token ids [web:19]
        self.assertEqual(y.dtype, torch.long)  # token ids [web:19]
        self.assertEqual(x.shape, (self.context_length,))  # 1D length T [web:19]
        self.assertEqual(y.shape, (self.context_length,))  # 1D length T [web:19]

    def test_shifted_targets(self):
        ds = LMDataset(self.text, self.tokenizer, self.context_length, self.stride)
        x, y = ds[0]
        # y is x shifted by one token to the right
        self.assertTrue(torch.equal(y[:-1], x[1:]))  # next-token prediction [web:19]
