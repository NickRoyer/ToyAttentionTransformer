import pickle
import os
import torch
import random

import numpy as np
import torch.nn as nn

class DataLoader(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.hasData = False
        self.dataset = dataset
        self.data_dir = os.path.join('data', dataset)        
        # attempt to derive vocab_size from the dataset
        self.meta_path = os.path.join(self.data_dir, 'meta.pkl')
        self.encoded_words = []
        self._init_data()
    
    def _init_data(self):
        if(self.verify_data()== False): 
            return
        
        with open(self.meta_path, 'rb') as f:
            self.meta = pickle.load(f)
            self.meta_vocab_size = self.meta['vocab_size']
            print(f"found vocab_size = {self.meta_vocab_size} (inside {self.meta_path})")
            self.stoi, self.itos = self.meta['stoi'], self.meta['itos']

    def verify_data(self):
        if(self.hasData == True): 
            return self.hasData

        # For simplicity just check if the meta file is present
        self.hasData = os.path.exists(self.meta_path)        
        if(self.hasData == False):
            print("WARNING: NO data has been prepared please run python data\\{self.dataset}\\prepare.py")
        
        return self.hasData
    
    def get_vocab_size(self):
        return self.meta_vocab_size

    def pad_to_four(self, input_tensor):
        """
        Ensures that every input tensor has exactly 4 integers by padding with zeros if necessary.
        :param input_tensor: 1D tensor with 1 to 4 integers
        :return: 1D tensor of length 4 with zero-padding
        """
        if (len(input_tensor) > 4):
            print('pad_to_four is limited to input of at most 4 chars')
            return
        
        max_length = 4  # Fixed output size
        padded_tensor = torch.zeros(max_length, dtype=torch.int64)  # Initialize with zeros

        # Copy input values into the padded tensor
        length = min(len(input_tensor), max_length)
        padded_tensor[:length] = input_tensor[:length]

        return padded_tensor

    def _load_toy_data(self):
        toy_data_path = os.path.join(self.data_dir, 'input.txt')        

        # Read the ASCII text file
        with open(toy_data_path, 'r', encoding='ascii') as f:
            text = f.read()

        # Split text into words (assuming space as the separator)
        words = text.split()        
        self.encoded_words = [self.pad_to_four(torch.tensor(self.encode(word), dtype=torch.int64)) for word in words]
        # the data came in a predictable method that interferes with training and validation
        random.shuffle(self.encoded_words)

        return self.encoded_words

    #Note; this is lazy and relies on prepare.py to have been run for vocabulary, if you edit the file then be aware you need to rerun that to get a proper vocab size.
    def get_toy_batch(self, batch_size, isTraining : bool = False):    
        if (len(self.encoded_words) == 0):
            self._load_toy_data()
        
        valStartIdx = int(len(self.encoded_words) * .8)
        valEndIdx = len(self.encoded_words) - 1
        startIdx = int(0 if isTraining == True else valStartIdx)
        endIdx = valStartIdx - 1 if isTraining == True else valEndIdx

        endIdx  = endIdx - batch_size

        #note if the data had less items the batch_size would overflow this
        ix = torch.randint(startIdx, endIdx, (1,))

        return self.encoded_words[ix: ix+ batch_size]

    # Get a traning and target from the data from the correct split (training or validation)
    def get_batch(self, batch_size: int, block_size: int, isTraining: bool, device = 'cpu'):
        if(self.verify_data() == False):
            return

        if (self.dataset == 'toy' and block_size == 4): # unique demo case to show word completion
            batch = self.get_toy_batch(batch_size)
            x = torch.stack([self.pad_to_four(item[:-1]) for item in batch] ) #use item[0 -> n -1] then pad the last element as 0
            y = torch.stack(batch)

            x, y = x.to(device), y.to(device)
            return x, y

        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if isTraining:
            data = np.memmap(os.path.join(self.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(self.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
                
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    def encode(self, input : list):
        if(self.verify_data() == False):
            return

        # This is how to encode the input char data into a tensor for feeding into the model, in this simple example its litterally the int representation of the char
        return [self.stoi[c] for c in input]

    def decode(self, input : list):
        if(self.verify_data() == False):
            return

        return ''.join([self.itos[i] for i in input]) #The inverse
