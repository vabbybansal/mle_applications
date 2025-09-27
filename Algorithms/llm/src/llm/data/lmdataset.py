from torch.utils.data import Dataset, DataLoader
import torch

class LMDataset(Dataset):
	"""
	Creates a language model training dataset from an input text corpus.
	- tokenizes based on the input tokenizer
	- creates input and target chunks where the targets are just inputs right shifted by one (next token prediction)
		- a,b,c -> b,c,d
	- adds an input stride between subsequent input rows. Ideally == context_length to prevent overfitting.
	"""
	def __init__(self, 
		txt, 
		tokenizer, 
		context_length, 
		stride):

		self.input_ids = []
		self.target_ids = []
		self.tokenizer = tokenizer
		self.context_length = context_length
		self.stride = stride

		# tokenize text
		token_ids = self.tokenize(txt)

		# create training dataset for the language model
		self.create_lm_data(token_ids)

	def tokenize(self, txt):
		# split the text into tokens using an already learnt encoder
		token_ids = self.tokenizer.encode(txt)
		assert len(token_ids) > self.context_length
		return token_ids

	def create_lm_data(self, token_ids):
		for i in range(0, len(token_ids) - self.context_length, self.stride):
			self.input_ids.append(torch.tensor(token_ids[i: i + self.context_length]))
			self.target_ids.append(torch.tensor(token_ids[i+1: i + self.context_length + 1]))

	def __len__(self):
		return len(self.input_ids)

	def __getitem__(self, idx):
		return self.input_ids[idx], self.target_ids[idx]