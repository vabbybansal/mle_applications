Tweet Sentiment Sentence Span Detection

ML Design
- Starting, Ending word detection (cross-entropy: each word position is a possible starting class and ending class (two outputs))
- Complete Span detection (Each word yes or no)
- Starting Ending index prediction (start and end words are sequence-indexed which is then predicted)

Others
- Using pre-trained Glove embeddings
- Using custom metric callback
- Standard LSTM Sequence Models
- NTLK tokenizers

Should be tried
- Using pre-trained RoBertA
- Basic Attention
- RoBertA + custom attention head
