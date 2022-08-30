from torch.utils.data import Dataset
import torch
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import preprocess as pre

class CodeDataset(Dataset):
    def __init__(self, data, tokenizer, vocab_size = 10000, max_len = 256):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.data = data
        self.name = self.column(self.data, 'number')
        self.inputs = self.column(self.data, 'desc')
        self.outputs = self.column(self.data, 'label')
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.outputs)
    
    def vocab(self):
        return self.vocab_size
    
    def column(self, data, tag1, tag2=None):
        if tag2:
            data_ = [row[tag1] + ' ' + row[tag2] for row in data]
        else:
            data_ = [row[tag1] for row in data]
        return data_
    
    def __getitem__(self, idx):
        name = self.name[idx]
        inputs = self.inputs[idx]
        outputs = self.outputs[idx]
        input_ids = self.tokenizer.encode(inputs)
        input_ids = np.array(input_ids)
        input_ids=input_ids.reshape(1,input_ids.shape[0])
        input_ids = pad_sequences(input_ids, self.max_len, padding='post')
        
        input_batch = torch.tensor(input_ids, dtype=torch.int64)
        target_batch = torch.tensor(outputs, dtype=torch.int64)
        return name, input_batch, target_batch
        
        
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    filepath='syzbot_data.pickle'
    train, valid, test = pre.load_data(filepath)
    
    train_dataset = CodeDataset(train)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    input_batch, target_batch = next(iter(train_dataloader))
    print(f"Feature batch shape: {input_batch.size()}")
    print(input_batch, target_batch)