class ToyConfig:
    block_size: int = 4
    n_layer: int = 1 # Number of layers within the transformer
    n_head: int = 1 # Number Attention heads
    n_embd: int = 12 # Number of C elements in the B, T, C Tensor that is used in the model ( B = Batch Size, T = Time position or Sequence Length, C = Number of embed elements )
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class ToyTrainingConfig:
    block_size : int = 0 # Need to set to ToyConfig.block_size
    device = 'cpu' # note in this sample toy code CUDA support is not tested 
    learning_rate : float = 1e-3 # this is relatively high 

#Sample config for shakespeare
# class ToyConfig:
#     block_size: int = 64
#     n_layer: int = 4 # Number of layers within the transformer
#     n_head: int = 2 # Number Attention heads
#     n_embd: int = 128 # Number of C elements in the B, T, C Tensor that is used in the model ( B = Batch Size, T = Time position or Sequence Length, C = Number of embed elements )
#     dropout: float = 0.0
#     bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

# class ToyTrainingConfig:
#     block_size : int = 0 # Need to set to ToyConfig.block_size
#     device = 'cpu' # note in this sample toy code CUDA support is not tested 
#     learning_rate : float = 1e-3 # this is relatively high 