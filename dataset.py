# This script is esigned to preprocess and format bilingual (source-target language) data for use in a transformer-based translation model.
# Its primary role is to prepare input sequences and masks in a way that is compatible with the model’s requirements for both the encoder and decoder inputs.
import torch
from torch.utils.data import Dataset

# This class essentially pre-processes the dataset from Huggingface to be formatted in such a way for the transformer to understand (creating tensors for the transformer). 
# This includes:
#   - Converting raw text sentences from the source and target languages into token IDs using the provided tokenizers.
#   - Adding special tokens such as <SOS> and <EOS> etc. 
#   - Adds masking to ensure model doesn't attend to other tokens more (using <PAD>)
class BilingualDataset(Dataset):

    # Constructor: 
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        
        # Initializing all values: 
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # Initializing special tokens [SOS, EOS, PAD] using pytorch method
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    # Helper function to return the length of the dataset from Huggingface
    def __len__(self):
        return len(self.ds)

    # This function fetches a specific item from the dataset and preprocesses it
    def __getitem__(self, idx):

        # Get the source/target pair from the dataset
        src_target_pair = self.ds[idx]

        # Get the text from both languages (source/target)
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        # Transform the text into token IDs using their respective tokenizers
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Calculates how many padding tokerns are needeed to reach the specified 'seq_len' for both the encoder and decoder tokens
        enc_num_padding_tokens = (self.seq_len - len(enc_input_tokens) - 2)     # We will add <s> and </s>, hence why -2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1       # We will only add <s>, and </s> only on the label, hence why -1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Creates the input seqeuence for the encoder by concatenating the SOS, tokenized sentence, EOS and Padding tokens till seq_len
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * enc_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # Creates the input seqeuence for the decoder by concatenating the SOS, tokenized sentence and Padding tokens till seq_len
        # The decoder input does not include the EOS token as it is inly added to the label (target output that model is trying to predict)
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # Creates label sequence by concatenating the tokenized target sequence, EOS token and padding tokens till it reaches seq_len
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # Return the tensors for the trasformer to train
        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)

            # Increasing the size of the encoder by adding paddings, hence a binary mask will ensure the extra tokens will not be seen by the self attention mechanism
            # Example: [1, 1, 1, ..., 1, 0, 0]  # '1' for non-padding, '0' for padding
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)

            # For decoder we have the binary mask and a special casual mask: to ensure that each position in the decoder can only attend to previous tokens (and not future tokens).
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # (1, seq_len) & (1, seq_len, seq_len),
            
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

# Casual mask for decoder to look at words which come before it and not look at words ahead of the sentence. 
# This function creates an upper triangular matrix by zeroing out all elements below the main diagonal from the top left to the bottom right of the matrix
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0 # Everything that is '0' is True
