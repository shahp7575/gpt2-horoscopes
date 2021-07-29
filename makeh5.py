import os
import config
import subprocess
import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import GPT2Tokenizer

# setup for h5 file
num_lines = subprocess.check_output(['wc', '-l', config.DATA_FILE])
num_lines = int(num_lines.split()[0])
print("Total lines in file:", num_lines)

# load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(config.MODEL_NAME)

# add special tokens to tokenizer
SPECIAL_TOKENS_DICT = {
    'pad_token': '<pad>',
    'additional_special_tokens': ['<|category|>', '<|horoscope|>']
}

tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
tokenizer.save_pretrained(os.path.join(config.MODEL_DIR, 'tokenizer'))

# grab these special tokens
category_token = tokenizer.additional_special_tokens_ids[0]
horoscope_token = tokenizer.additional_special_tokens_ids[1]
pad_token = tokenizer.pad_token_id
eos_token = tokenizer.eos_token_id

# create h5 file
dt = h5py.special_dtype(vlen=np.dtype('int32'))

with h5py.File(os.path.join(config.DATA_DIR, 'horoscope_cleaned.h5'), 'w') as h5f:
    
    dset1 = h5f.create_dataset('texts',
                               shape=(num_lines,),
                               compression=None,
                               dtype=dt)
    dset2 = h5f.create_dataset('labels',
                               shape=(num_lines,),
                               compression=None,
                               dtype=dt)
    dset3 = h5f.create_dataset('segments',
                               shape=(num_lines,),
                               compression=None,
                               dtype=dt)
    
    for i in tqdm(range(0, num_lines, config.H5_CHUNKSIZE)):
        
        df = pd.read_csv(config.DATA_FILE, nrows=config.H5_CHUNKSIZE, skiprows=i, header=None)
        df_len = df.shape[0]
        
        tokens_list = []
        labels_list = []
        segments_list = []
        count = 0
        
        for idx, item in enumerate(df.values):
            # features
            category = [category_token] + tokenizer.encode(item[0], max_length=6-1)
            horoscope = [horoscope_token] + tokenizer.encode(item[1], max_length=config.SEQUENCE_LENGTH-5-2) + [eos_token]
            
            tokens = category + horoscope + [pad_token]*(config.SEQUENCE_LENGTH - len(category) - len(horoscope))
            segments = [category_token] * len(category) + [horoscope_token]*(config.SEQUENCE_LENGTH - len(category))
            labels = [-100]*(len(category)+1) + horoscope[1:] + [-100]*(config.SEQUENCE_LENGTH - len(category) - len(horoscope))
            
            assert(len(tokens)==config.SEQUENCE_LENGTH)
            assert(len(segments)==config.SEQUENCE_LENGTH)
            assert(len(labels)==config.SEQUENCE_LENGTH)
            
            tokens = np.asarray(tokens, dtype=np.int32)
            segments = np.asarray(segments, dtype=np.int32)
            labels = np.asarray(labels, dtype=np.int32)
            
            tokens_list.append(tokens)
            labels_list.append(labels)
            segments_list.append(segments)
            
        dset1[i:i+df_len] = tokens_list
        dset2[i:i+df_len] = labels_list
        dset3[i:i+df_len] = segments_list