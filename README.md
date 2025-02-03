This is a sample pytorch implementation of an attention transformer for an LLM. This sample is adapted from the work provided here: https://github.com/karpathy/nanoGPT. 

For performance optimzation and added complexity see the original work, this is intended to show a simplified training loop that is accessible via the notebook.

The main test case here is 1 -> 4 letter words. The model is intended to learn the different words available and predict the next character in the word sequence. For purposes of keeping this simple the vocabulary of the input has been reduced to 10 characters and the nonce char (space in this case)

I highly recommend you follow along on the videos here: https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ

and further invetigate the models / architectures here: https://github.com/huggingface/transformers/tree/v4.48.2/src/transformers

In the mean time I hope this notebook (https://github.com/NickRoyer/ToyAttentionTransformer/blob/main/TrainingLoop.ipynb) can help you visualize a simple transformer training loop

The main logic is split:
ModelDetails => Contains the details of how the model is created including an attention architecture
Model => the bones of the model 
Trainer => impleentation of a simple training loop
config => standard configuration

NOTE: There are two main samples provided: the "toy" example which shows a 1 -> 4 letter word completion and a more complex example of trying to complete the shakespeare text.
