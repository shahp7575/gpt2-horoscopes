import os
import gc
import random
import config
import pickle
import dataset
from tqdm.auto import tqdm
import math
import h5py
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def train_valid_loaders(dataset, valid_fraction=0.1, **kwargs):
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(math.floor(valid_fraction*num_train))
    
    if not('shuffle' in kwargs and not kwargs['shuffle']):
        np.random.shuffle(indices)
    if 'num_workers' not in kwargs:
        kwargs['num_workers'] = 1
        
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    train_loader = torch.utils.data.DataLoader(dataset,
                                               sampler=train_sampler,
                                               **kwargs)
    valid_loader = torch.utils.data.DataLoader(dataset,
                                               sampler=valid_sampler,
                                               **kwargs)
    
    return train_loader, valid_loader

def fit(model, optimizer, scheduler, train_dl, val_dl, epochs=1, sample_every=100, device=torch.device('cpu')):
    
    training_stats = []
    best_eval_loss = 99999
    
    for i in range(epochs):

        print('\n--- Starting epoch #{} ---'.format(i))
        print("Training...")
        print("")

        model.train()

        # These 2 lists will keep track of the batch losses and batch sizes over one epoch:
        losses = []
        total_train_loss = 0
        model.train()
        
        for step, batch in enumerate(tqdm(train_dl, desc="Training")):
            # Move the batch to the training device:
            texts = batch[0].to(device)
            labs = batch[1].to(device)
            segs = batch[2].to(device)

            # Call the model with the token ids, segment ids, and the ground truth (labels)
            outputs = model(texts, labels=labs, token_type_ids=segs)

            # Add the loss and batch size to the list:
            loss = outputs[0]
            batch_loss = loss.item()
            total_train_loss += batch_loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        # Compute the average cost over one epoch:
        avg_train_loss = total_train_loss / len(train_dl)

        # Now do the same thing for validation:
        print("\n")
        print("Validation...")
        print("")
        
        model.eval()

        total_eval_loss = 0

        with torch.no_grad():
            losses = []

            for step, batch in enumerate(tqdm(val_dl, desc="Validation")):
                texts = batch[0].to(device)
                labs = batch[1].to(device)
                segs = batch[2].to(device)
                outputs = model(texts, labels=labs, token_type_ids=segs)
                batch_loss = loss.item()
                total_eval_loss += batch_loss

        avg_val_loss = total_eval_loss / len(val_dl)
        
        # Save model
        if avg_val_loss < best_eval_loss:
            
            print("\nSaving model...")
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(config.MODEL_DIR)
            print("Model saved.")
            best_eval_loss = avg_val_loss

        training_stats.append({'epoch': i + 1,
                               'Training loss': avg_train_loss,
                               'Validation loss': avg_val_loss})
        
        print('\n--- Epoch #{} finished --- Training loss: {} / Validation loss: {}'.format(i+1, avg_train_loss, avg_val_loss))
        
    # save training stats to pickle file
    with open(os.path.join(config.MODEL_DIR, 'training_stats.pkl'), 'wb') as f:
        pickle.dump(training_stats, f)
        

if __name__ == "__main__":
    
    # set seeds
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    train_loader, valid_loader = train_valid_loaders(dataset.HoroscopeDataset(os.path.join(config.DATA_DIR, 'horoscope_cleaned.h5'), 'texts', 'labels', 'segments'),
                                                     valid_fraction=0.1,
                                                     batch_size=4,
                                                     num_workers=2)
    
    tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(config.MODEL_DIR, 'tokenizer'))
    model = GPT2LMHeadModel.from_pretrained(config.MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    
    # move the model to the GPU
    device = torch.device(config.DEVICE)
    model.to(device)
    
    # optimizer
    optimizer = AdamW(model.parameters(),
                      lr=config.LEARNING_RATE,
                      eps=config.EPSILON)
    
    total_steps = len(train_loader) * config.EPOCHS
    
    # scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=config.WARMUP_STEPS,
                                                num_training_steps=total_steps)
    
    # fine-tune gpt2
    optimizer = AdamW(model.parameters())
    fit(model, optimizer, scheduler, train_loader, valid_loader, epochs=config.EPOCHS, device=config.DEVICE)
    
    
    
