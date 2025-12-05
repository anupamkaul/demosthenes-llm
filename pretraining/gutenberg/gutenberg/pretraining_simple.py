# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
Script for pretraining a small GPT-2 124M parameter model
on books from Project Gutenberg.

Before running this script, make sure you downloaded and
processed the dataset as described in the README.md.
"""

import argparse
import os
from pathlib import Path
import time
import tiktoken
import torch

'''
from llms_from_scratch.ch02 import create_dataloader_v1
from llms_from_scratch.ch04 import GPTModel, generate_and_print_sample
from llms_from_scratch.ch05 import calc_loss_batch, evaluate_model, plot_losses
'''

import sys, os
sys.path.append( os.path.join( os.path.dirname(os.path.abspath(__file__)),  '../../../tokenizers/') )
sys.path.append( os.path.join( os.path.dirname(os.path.abspath(__file__)),  '../../../attention/') )
sys.path.append( os.path.join( os.path.dirname(os.path.abspath(__file__)),  '../../../llm-infra/') )
sys.path.append( os.path.join( os.path.dirname(os.path.abspath(__file__)),  '../../../pretraining') )

from GPTModel import GPTModel  
from dataloaderV1 import create_dataloader_v1
from utils_loss import calc_loss_batch, evaluate_model, generate_and_print_sample, plot_losses


def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    return text_data


def create_dataloaders(text_data, train_ratio, batch_size, max_length, stride, num_workers=0):
    split_idx = int(train_ratio * len(text_data))
    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=True,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=False,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, val_loader


def convert_time(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return int(hours), int(minutes), int(seconds)


def print_eta(start_time, book_start_time, index, total_files):
    book_end_time = time.time()  # End time of processing this book
    elapsed_time = book_end_time - book_start_time
    total_elapsed_time = book_end_time - start_time
    books_remaining = total_files - index
    average_time_per_book = total_elapsed_time / index
    eta = average_time_per_book * books_remaining

    book_h, book_m, book_s = convert_time(elapsed_time)
    total_h, total_m, total_s = convert_time(total_elapsed_time)
    eta_h, eta_m, eta_s = convert_time(eta)

    print(f"Book processed {book_h}h {book_m}m {book_s}s"
          f"\nTotal time elapsed {total_h}h {total_m}m {total_s}s"
          f"\nETA for remaining books: {eta_h}h {eta_m}m {eta_s}s")

import json
def save_training_state(n_epochs, file_enum, input_batch_counter, tokens_seen, global_step, filename="training_state.json"):
    """ save training state to json, to resume an interrupted training for the LLM """
    
    state = {
        "n_epochs": n_epochs,
        "file_enum": file_enum,
        "input_batch_counter": input_batch_counter,
        "tokens_seen" : tokens_seen,
        "global_step": global_step
    }
    with open(filename, 'w') as f:
        json.dump(state, f) 
    print(f"Training state saved for epoch {n_epochs} file index {file_enum} batch_counter  {input_batch_counter} tokens {tokens_seen} global_step {global_step}")


# training state is saved in the following variables
# these are global vars, since I modify them inside the training loop
# (another python kludge, and positionally I need to declare it prior to its usage)

sv_n_epochs = 0            # for epochs
sv_start_file_enum = 1     # for file index (every 500MB clubbed text file) 
sv_input_batch_counter = 1   # for current input and target batches
sv_tokens_seen = 0         # for token_seen
sv_global_step = 0         # the global step var in the training loop

def train_model_simple(model, optimizer, device, n_epochs,
                       eval_freq, eval_iter, print_sample_iter, start_context,
                       output_dir, save_ckpt_freq, tokenizer,
                       batch_size=2, train_ratio=0.90):

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = 0 
    start_time = time.time()

    # batch_size of 4 is about 38K global_step iterations (to cover the entirety of training set data, via sliding window)
    # this hack is for shortening the data seen, albiet at the cost of accuracy. But I will get a fuller picture from all of the books
    # in a shorter amount of time (i.e. training should complete in number_epochs * max_eval_limit * number of new-indexed books)
    # additional TODO is to save the state of files, epochs, and actual input batches to prevent biasing when restarting the program

    max_eval_limit = 1000
    #max_eval_limit = 3 # for simulation

    global sv_global_step
    global_step = sv_global_step

    global sv_tokens_seen
    tokens_seen = sv_tokens_seen

    try:
        for epoch in range(n_epochs):

            global sv_n_epochs
            if (epoch < sv_n_epochs):
                continue

            print("\ntraining for epoch ", epoch, " of ", n_epochs, "\n")
            print("batch size ", batch_size)
            input()

            # Iterate over the books in the training corpus
            for index, file_path in enumerate(all_files, 1):

                # upon interruption, we save index in the following loop and
                # we restore it as sv_start_file_enum. We skip the enumeration
                # via continue till we land on the right book index

                # enumerate with start_index just moves the start index but retains the
                # full iterable list (which is not what we're looking for). So we will use
                # sv_ counter to skip to the actual iterable, while still starting the enumeration
                # as 1. (This is a basic limitation of python's enumeration construct)

                global sv_start_file_enum
                if (index < sv_start_file_enum):
                    continue

                print("new index: ", index, "file path: ", file_path, "<ENTER>")

                # need these vars to save and restore training states when interrupted

                global sv_input_batch_counter
                # first time interrupts (on file read, tokenization) should preserve saved batch states
                if index == sv_start_file_enum: 
                    input_batch_counter = sv_input_batch_counter
                else:
                    input_batch_counter = 0

                '''
                # training loop simulation aid
                input()
                continue
                '''

                print(f"Reading and splitting file {index} of {total_files}: {file_path} into a {train_ratio} split between train and validation")

                book_start_time = time.time()
                text_data = read_text_file(file_path) + " <|endoftext|> "

                print(f"Tokenizing file {index} of {total_files}: {file_path}")

                # tokenization happens in create_dataloaders, which calls create_dataloader_v1
                # (defined in tokenizers/dataloaderV1.py) which calls GPTDatasetV1
                # (defined in tokenizers/GPTDatsetV1.py) and this class (GPTDatasetV1) tokenizes the text

                # Initialize new data loaders for each book
                # 90% of this file is for training and 10% is retained for validation

                train_loader, val_loader = create_dataloaders(
                    text_data,
                    train_ratio=train_ratio,
                    batch_size=batch_size,
                    max_length=GPT_CONFIG_124M["context_length"],
                    stride=GPT_CONFIG_124M["context_length"],
                    num_workers=0
                )

                '''
                print("Train loader:") #x, y are input_batch and target_batch..
                for x,y in train_loader:
                    print(x.shape, y.shape)
                '''

                # for input, target batches, since they increase sequentially by
                # 1 iterator each time (they are basically different by 1, and both
                # have the same stride window), we will save a counter and then skip
                # the next for loop by the same counter (saved as a sv_input_batch_counter)
                # by using continue. That should bring us to the saved state and restore
                # the for loop as if nothing happened. Its the same iterator principle, and
                # we don't need to care "what" the iterator setup is ; the for loop is already
                # doing that for us. This "counter" will always be less than either the full
                # length of the loop or less than max_iter. In fact I can get rid of max_iter
                # as well and do an actual training that is saved/restored...

                print("\nTraining ...")
                model.train()  # set up training params

                # training loop

                input_batch_counter = 0
                for input_batch, target_batch in train_loader:
                    input_batch_counter += 1

                    # a good debug to indicate how long this loop will run [ len(train_loader} ]
                    # print("\ndebug: len input_batch: ", len(input_batch), "len target_batch: ", len(target_batch), "len train_loader: ", len(train_loader))

                    # we will need to save this input_batch, target_batch iterations and restart
                    # without losing context, post an interruption. How big are these anyways?

                    if (input_batch_counter < sv_input_batch_counter):
                        continue

                    print(input_batch_counter, " input batch: ", input_batch, "\ntarget batch : ", target_batch)

                    '''
                    # training loop simulation aid
                    input()
                    continue
                    '''

                    print("optimizer setting")
                    optimizer.zero_grad()
                    print("optimizer set")

                    print("loss calculating")
                    loss = calc_loss_batch(input_batch, target_batch, model, device)
                    print("loss calculated")

                    print("backprop start")
                    loss.backward()
                    print("backprop end")

                    print("optimizer step start")
                    optimizer.step()
                    print("optimizer step end")

                    tokens_seen += input_batch.numel()
                    print("we have tokens_seen")

                    global_step += 1
                    print("global step: ", global_step, " tokens seen: ", tokens_seen)

                    # Optional evaluation step
                    if global_step % eval_freq == 0:
                        print("evaluating model, noting train + validation loss (interim)")
                        train_loss, val_loss = evaluate_model(
                            model, train_loader, val_loader, device, eval_iter)
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                        track_tokens_seen.append(tokens_seen)
                        print(f"Ep {epoch+1} (Step {global_step}): "
                              f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

                    # Generate text passage
                    if global_step % print_sample_iter == 0:

                        print("\ngenerate and print sample..")
                        generate_and_print_sample(
                            model, tokenizer, device, start_context
                        )

                    # if I am using colab I don't need this limit check
                    if global_step % max_eval_limit == 0:

                        print("reached max eval limit, moving to next book\n")

                        print("\ngenerate and print sample..")
                        generate_and_print_sample(
                            model, tokenizer, device, start_context
                        )

                        break # skip the rest of the iterations and exit loop

                print("\nout of input_batch inner for loop\n")

                if global_step % save_ckpt_freq:

                    print("saving model (interim)")
                    file_name = output_dir / f"model_and_optmzr_pg_{global_step}.pth"

                    #torch.save(model.state_dict(), file_name)
                    # save both the model and the optimizer states
                    torch.save( 
                         {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                         }, 
                         file_name
                    )

                    # save the model
                    model_file_name = output_dir / "model_and_optmzr_pg_final.pth"
                    #torch.save(model.state_dict(), model_file_name)
                    torch.save( 
                         {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                         }, 
                         model_file_name
                    )

                    print(f"Saved {file_name}")
                    print(f"Saved {model_file_name}")

                print("some stats: ")
                print_eta(start_time, book_start_time, index, total_files)

            # this is the end of the first primary loop (epoch). Here is where
            # index has to be zero (not anywhere else) to synchronize with the
            # saved states and to prevent newer epochs from starting with the 
            # maxed out index. Index is naturally going to be recalculated to zero
            # following the flow mechanics. This is either where we save and load
            # again or initialize the saved enum to zero

            print("switching to next epoch for training")
            sv_start_file_enum = 1

        # this is where the final for loop ends for training
        print("END OF TRAINING")
        # save the model
        model_file_name = output_dir / "model_pg_final.pth"
        torch.save(model.state_dict(), model_file_name)
        print(f"Saved {model_file_name}")

        # save in-progress training state
        save_training_state(epoch, index, input_batch_counter, tokens_seen, global_step)
        print(f"Saved training state")


    except KeyboardInterrupt:

        file_name = output_dir / f"model_and_optmzr_pg_{global_step}_interrupted.pth"
        #torch.save(model.state_dict(), file_name)
        
        torch.save( 
             {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
             }, 
             file_name
        )

        # save the model
        model_file_name = output_dir / "model_and_optmzr_pg_final.pth"
        #torch.save(model.state_dict(), model_file_name)
        torch.save( 
             {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
             }, 
             model_file_name
        )

        print(f"Saved {file_name} and {model_file_name}")

        # save in-progress training state
        save_training_state(epoch, index, input_batch_counter, tokens_seen, global_step)
        print(f"Saved training state")

    return train_losses, val_losses, track_tokens_seen

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GPT Model Training Configuration')

    # modified these training params for a CPU friendly debug
    parser.add_argument('--data_dir', type=str, default='data/combined-80mb',
    #parser.add_argument('--data_dir', type=str, default='data/preprocessed.0',
                        help='Directory containing the training data')
    parser.add_argument('--output_dir', type=str, default='model_checkpoints',
                        help='Directory where the model checkpoints will be saved')
    parser.add_argument('--n_epochs', type=int, default=3,
                        help='Number of epochs to train the model')
    parser.add_argument('--print_sample_iter', type=int, default=5,
                        help='Iterations between printing sample outputs')
    parser.add_argument('--eval_freq', type=int, default=5,
                        help='Frequency of evaluations during training')
    parser.add_argument('--save_ckpt_freq', type=int, default=10,
                        help='Frequency of saving model checkpoints during training')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=2,
    #parser.add_argument('--batch_size', type=int, default=256,
    #parser.add_argument('--batch_size', type=int, default=1024, # override here (mem resource limit reached)
                        help='Batch size for training')
    parser.add_argument('--debug', type=bool, default=False,
                        help='Uses a very small model for debugging purposes')

    args = parser.parse_args()
    print("pretraining args: ", args)

    if args.debug:
        GPT_CONFIG_124M = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 10,    # Context length
            "emb_dim": 12,           # Embedding dimension
            "n_heads": 2,            # Number of attention heads
            "n_layers": 2,           # Number of layers
            "drop_rate": 0.0,        # Dropout rate, deactivated via 0.0 as dropout in LLMs is not recommended anymore
            "qkv_bias": False        # Query-key-value bias
        }

    else:
        GPT_CONFIG_124M = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 1024,  # Context length
            "emb_dim": 768,          # Embedding dimension
            "n_heads": 12,           # Number of attention heads
            "n_layers": 12,          # Number of layers
            "drop_rate": 0.1,        # Dropout rate
            "qkv_bias": False        # Query-key-value bias
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device for training: ", device)

    # https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/
    print("is MPS available: ", torch.backends.mps.is_available())

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    # an unfortunate hack for my local ubuntu 22.04 : even when cuda force it to CPU
    # (this is because memory requests to my GPU0 exceed its total mem available, needs debugging)

    device = torch.device("cpu")
    print("device override (for my local ubuntu): ", device)

    # first load up the training (checkpoint) states to continue a previously interrupted training

    try:
        with open("training_state.json", 'r') as f:
            saved_state = json.load(f)
            print("loading training state: ", saved_state)
            # load the variables
            sv_n_epochs           = saved_state["n_epochs"]
            sv_start_file_enum    = saved_state["file_enum"]
            sv_input_batch_counter= saved_state["input_batch_counter"]
            sv_tokens_seen        = saved_state["tokens_seen"]
            sv_global_step        = saved_state["global_step"]

    except FileNotFoundError:
            print("No training saved states ! Start training from the beginning")
            sv_n_epochs=0
            sv_start_file_enum=1
            sv_input_batch_counter=1
            sv_tokens_seen=0
            sv_global_step=0

    # next, load previously saved instance of the model (to continue training) and the training states
    # enable this once I save it first time (I don't know the format, so let the output drive the input)

    try:

        checkpoint = torch.load("model_checkpoints/model_and_optmzr_pg_final.pth", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print("loaded previously saved model and optimizer to continue training..")

    except FileNotFoundError:
        print("model not found on disk. monitor as a one time thing, error out if repeats")

        # Re-initialize training states here : case of moving machines when data is not exported
        # but previous states from running on another machine was checked-in

        # (using files: if there is a previous state cache then re-initialize it and save the previous copy
        # but since I reverted the lookup order (first load training states) hence reinit of variables will
        # do the trick here nicely. The assumption is that every run that is interrupted will update and save
        # both the model and the checkpoint states file)

        print("No training saved states ! Start training from the beginning")
        sv_n_epochs=0
        sv_start_file_enum=1
        sv_input_batch_counter=1
        sv_tokens_seen=0
        sv_global_step=0


    model.to(device)

    # still need this init in case a saved checkpoint (or model) doesn't exist
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    tokenizer = tiktoken.get_encoding("gpt2")

    data_dir = args.data_dir
    all_files = [os.path.join(path, name) for path, subdirs, files
                 in os.walk(data_dir) for name in files if name.endswith((".txt"))]
    total_files = len(all_files)

    if total_files == 0:
        print("No training text files found. Make sure you "
              "selected the correct input directory")
        quit()
    print("Total files for training:", total_files)
    print("Files:\n", all_files)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input("Ready to commence training! <enter>")

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, optimizer, device,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        eval_iter=1,
        print_sample_iter=args.print_sample_iter,
        output_dir=output_dir,
        save_ckpt_freq=args.save_ckpt_freq,
        start_context="Every effort moves you", # once trained this can be user input
        tokenizer=tokenizer
    )

    # these are probably worth saving as well, for plotting (arrays)
    print("tokens_seen[] = ", tokens_seen)
    print("train_losses[] = ", train_losses)
    print("val_losses[] = ", val_losses)

    epochs_tensor = torch.linspace(0, args.n_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    # save the model
    torch.save(model.state_dict(), output_dir / "model_pg_final.pth")
    print(f"Maximum GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
