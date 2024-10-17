# This script is for extracting data and tokenizing it appropriately (Word2Vec - Splitting it by words)
# Then we will implement training the transformer with the tokenized input sequence.
# Dataset Link: https://huggingface.co/datasets/Helsinki-NLP/opus_books
# --------------------------------------------------------------------------------------------------------
# Project Directory Imports:
from model import build_transformer
from config import get_config, get_weights_file_path, latest_weights_file_path
from dataset import BilingualDataset, causal_mask

# Torch Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchmetrics

# Hugging face datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

# General Imports
from pathlib import Path
import warnings

# --------------------------------------------------------------------------------------------------------
#                                       Pre-processing the Dataset:
# --------------------------------------------------------------------------------------------------------
# For each item in the dataset, we extract the sentence in a specified language.
def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]

# This will build our tokenizer from scratch with a particular configuration, dataset and langauge
# This will convert our input sequence into tokens or words for the transformer to understand
def get_or_build_tokenizer(config, ds, lang): 
    tokenizer_path = Path(config["tokenizer_file"].format(lang))

    # If the file path doesn't already exists
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour

        # The MAIN tokenizer which uses WordLevel model maps entire words to unique IDs
        # Any out words not seen during training are specified with UNK (unknown)
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))

        # The Whitespace pre-tokenizer is used to split the input text on whitespace - treating each word as a token
        tokenizer.pre_tokenizer = Whitespace()

        # Object to train the tokenizer - with special tokens (unknown, padding, Start of Seqeuence, End of Sequence)
        trainer = WordLevelTrainer(
                special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
            )

        # Trains the tokenizer using the extracted sentences and saves it into the tokenizer path 
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))

    # Otherwise, we use the tokenizer from the tokenizer file path
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

# This function aims to retrieve the dataset from HuggingFace and split and process it for the transformer (adding tensors for it)
def get_ds(config): 
    # Extracting the entire 'opus_books' training dataset from HuggingFace which contains src language and target language
    # The dataset is multilinguial hence we can specify the source and target language from our configurations.
    ds_raw = load_dataset(
        f"{config['datasource']}",
        f"{config['lang_src']}-{config['lang_tgt']}",
        split="train",
    )

    # Build tokenizers for our source and target languages from the raw datasets
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # We split the data into 90% - training, 10% - cross_validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    # Pass the split training datasets into the BillingualDataset class for initializing of pre-processing the data into tokens
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    # Create the Data Loaders which batches and shuffles the dataset instead of passing it one by one for training and evaluation.
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

# This function sets up and gets our transformer model that we created from the model.py script.
def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

# --------------------------------------------------------------------------------------------------------
#                                             Validation Loop :
# --------------------------------------------------------------------------------------------------------
# This function is responsible for generating the predicted sequence from the model using a greedy decoding strategy. 
# In greedy decoding, the model always selects the token with the highest probability at each time step. 
# This process is repeated until the model generates an end-of-sequence token (<EOS>) or the maximum length (max_len) is reached.
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)

    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    # Loop runs until decoder generates EOS or sequence reaches max length. 
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

# This is an OPTIONAL function loop to run validation for evaluating the model on a validation dataset throughout the training phase.
# This is done by calling greedy decode to generate predictions adn comparing them with the actual target sentence
def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    # Model is in evaluation mode
    model.eval()
    
    count = 0
    source_texts = []
    expected = []
    predicted = []

    # Try to get the console window width and if not then we use 80 as deault. 
    console_width = 80

    # Since we are performing validation, we don't need to calculate gradients
    with torch.no_grad():

        # For each batch in the validation dataset
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)   # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)     # (b, 1, 1, seq_len)

            # Check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            # Generate the prediction for the current sentence using the greedy decode function
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            # Retrieve the source text and target text for the current example and store it for later evaluation
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{'SOURCE: ':>12}{source_text}")
            print_msg(f"{'TARGET: ':>12}{target_text}")
            print_msg(f"{'PREDICTED: ':>12}{model_out_text}")

            # Stop after some value to avoid printing too many results
            if count == num_examples:
                print_msg('-'*console_width)
                break
        
        # If the tensorboard writer is provided, then we can calculate the appropriate metrics
        if writer:
            # Evaluate the character error rate
            # Compute the char error rate 
            metric = torchmetrics.CharErrorRate()
            cer = metric(predicted, expected)
            writer.add_scalar('validation cer', cer, global_step)
            writer.flush()

            # Compute the word error rate
            metric = torchmetrics.WordErrorRate()
            wer = metric(predicted, expected)
            writer.add_scalar('validation wer', wer, global_step)
            writer.flush()

            # Compute the BLEU metric
            metric = torchmetrics.BLEUScore()
            bleu = metric(predicted, expected)
            writer.add_scalar('validation BLEU', bleu, global_step)
            writer.flush()

# --------------------------------------------------------------------------------------------------------
#                                             Training Loop :
# --------------------------------------------------------------------------------------------------------
# This is the training loop in which we will train the transformer based on our pre-processed dataset
def train_model(config): 
    # Define the device and print information about the hardware device that PyTorch will use for training and evaluation 
    # Preferred if CUDA (NVIDIA GPU) or MPS (Apple M1/M2) otherwise model will be training using CPU
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    
    elif (device == 'mps'):
        print("Device name: <mps>")
    
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    # Get the dataset and retrieve the dataloaders and tokenizers
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    # Get the model that we will eventually train on 
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard will allow us to visualize the loss
    writer = SummaryWriter(config['experiment_name'])

    # Creating the optimizer for gradient descent 
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    
    else:
        print('No model to preload, starting from scratch')

    # Loss Function used (we don't want padding tokens to contribute to the loss)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # Training loop: 
    for epoch in range(initial_epoch, config['num_epochs']):
        
        # Train the model and wrapping the train_dataloader batches using tqdm which will show nice progress bar
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        
        # For each batch 
        for batch in batch_iterator:
            # Get the tensors 
            encoder_input = batch['encoder_input'].to(device)   # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device)   # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)     # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)     # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask)                                  # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)    # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)                                                 # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch (OPTIONAL)
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

# Remove warnings and train the model based on configurations
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)