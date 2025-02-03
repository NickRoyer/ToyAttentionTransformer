import torch
import torch.nn as nn

import time
from config import ToyTrainingConfig
from toy_model import ToyModel
from data_loader import DataLoader

class ToyTrainer(nn.Module):
    def __init__(self, config : ToyTrainingConfig, model : ToyModel, data : DataLoader):
        super().__init__()
        self.config = config
        self.model = model
        self.data = data
            
        # optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # For presentation only
    def get_single_batch(self):
        return self.data.get_batch(batch_size=1, block_size=self.config.block_size, isTraining=False) # fetch the very first batch

    def TrainingLoop( self, batch_size: int, max_iters : int, log_interval: int = 0 ):
        X, Y = self.data.get_batch(batch_size=batch_size, block_size=self.config.block_size, isTraining=True) # fetch the very first batch
        t0 = time.time()
        iter_num = 0 # number of iterations in the lifetime of this process

        while True:        
            X, Y = self.data.get_batch(batch_size=batch_size, block_size=self.config.block_size, isTraining=True)
            _, loss = self.model(X, Y)
                
            # backward pass
            loss.backward()
        
            self.optimizer.step()

            # flush the gradients as soon as we can, no need for this memory anymore
            self.optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if log_interval > 0 and iter_num % log_interval == 0:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item()
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
            
            iter_num += 1

            # termination conditions
            if iter_num > max_iters:
                break