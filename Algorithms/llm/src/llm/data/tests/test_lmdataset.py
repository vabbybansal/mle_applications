import unittest
from llm.data.lmdataset import LMDataset

class TiktokenTokenizer:
    """Thin wrapper around tiktoken's encoding."""
    def __init__(self, encoding_name: str = "cl100k_base"):
        import tiktoken
        self.enc = tiktoken.get_encoding(encoding_name)

    def encode(self, txt: str):
        return self.enc.encode(txt)

class TestLMDataset(unittest.TestCase):
	def setUp(self):
		self.text = "hello world, this is a test case"
		self.tokenizer = TiktokenTokenizer("cl100k_base")
		self.context_length = 5
		self.stride = 5

	def test_shifted_targets(self):
		ds = LMDataset(
				self.text,
				self.tokenizer,
				self.context_length,
				self.stride
			)
		print(len(ds))