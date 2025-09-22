
from llm.data.lmdataset import LMDataset

from torch.utils.data import DataLoader
import tiktoken

class LMDataLoader():
	@staticmethod
	def get_dataloader(text, context_length, stride, batch_size, tiktoken_encoding_type, num_workers, shuffle=False, drop_last=True):

		tokenizer = tiktoken.get_encoding(tiktoken_encoding_type)

		dataset = LMDataset(
					txt = text,
					tokenizer=tokenizer,
					context_length=context_length,
					stride=stride
			)

		dataloader = DataLoader(
						dataset=dataset,
						batch_size=batch_size,
						shuffle=shuffle,
						drop_last=drop_last,
						num_workers=num_workers
					)

		return dataloader