import unittest
from llm.data.lmdataloader import LMDataLoader
import torch
from torch.utils.data import DataLoader


class TestLMDataLoader(unittest.TestCase):

	def test_get_dataloader(self):
		

		dl = LMDataLoader.get_dataloader(
				text="hello world, this is a test case, jhgsjhags sh hd  dhjjsh sj hs j hsjh hsj hjsh h jhsj ", 
				context_length=3, 
				stride=2, 
				batch_size=2, 
				tiktoken_encoding_type="cl100k_base", 
				num_workers=2,
				shuffle=False, 
				drop_last=True
			)

		self.assertIsInstance(dl, DataLoader)

		X,Y = next(iter(dl))

		self.assertIsInstance(X, torch.Tensor)
		self.assertEqual(X.shape[0], 2)
		self.assertEqual(X.shape[1], 3)

