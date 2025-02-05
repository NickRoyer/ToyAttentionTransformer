# Toy Attention Transformer (PyTorch Implementation)

This is a **sample PyTorch implementation** of an **attention-based transformer** for a **large language model (LLM)**. This example is adapted from the excellent work found here:  
🔗 **[nanoGPT by Karpathy](https://github.com/karpathy/nanoGPT)**  

For **performance optimizations** and **additional complexity**, refer to the original work. This implementation is **designed to demonstrate a simplified training loop** that is accessible via the provided Jupyter Notebook.

---

## Overview

The primary test case here involves **1 to 4-letter words**. The model is designed to **learn the available words** and **predict the next character in a word sequence**.  

To **keep the implementation simple**, the input vocabulary has been **limited to 10 characters** plus a **nonce character (space in this case)**.

For a **deeper understanding**, I highly recommend following along with this **video series**:  
🎥 **[YouTube - Transformer Models](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)**  

Additionally, you can **explore more models and architectures** here:  
🔗 **[Hugging Face Transformers (v4.48.2)](https://github.com/huggingface/transformers/tree/v4.48.2/src/transformers)**  

---

## How to Use

In the meantime, I hope this **notebook** helps you visualize a **simple transformer training loop**:  
📌 **[TrainingLoop.ipynb](https://github.com/NickRoyer/ToyAttentionTransformer/blob/main/TrainingLoop.ipynb)**  

After running the init, and the training loop

**Get Target:** (sample output, IMO 4 letter targets are better)

T  : Target single Character according to source input 

Input: 
 COL 
Target: 
 COLT

X: 
 tensor([[2, 7, 5, 0]])
Y: 
 tensor([[ 2,  7,  5, 10]])

**Single Model loop:** (sample output)
Results of the Model vs the Target input: **Note: rerunning this will do a backwards pass and thus improve the output 

C -> ["' ':0.1%", "'A':0.1%", "'C':98.6%", "'E':0.1%", "'I':0.2%", "'L':0.1%", "'N':0.3%", "'O':0.1%", "'R':0.1%", "'S':0.0%", "'T':0.2%"]

O -> ["' ':0.1%", "'A':0.1%", "'C':0.1%", "'E':0.1%", "'I':0.1%", "'L':0.1%", "'N':0.1%", "'O':98.8%", "'R':0.1%", "'S':0.1%", "'T':0.3%"]

L -> ["' ':0.2%", "'A':0.2%", "'C':0.1%", "'E':0.0%", "'I':0.0%", "'L':98.8%", "'N':0.1%", "'O':0.1%", "'R':0.1%", "'S':0.2%", "'T':0.1%"]

T -> ["' ':0.0%", "'A':8.3%", "'C':2.4%", "'E':2.3%", "'I':0.6%", "'L':1.4%", "'N':3.1%", "'O':5.2%", "'R':3.0%", "'S':15.7%", "'T':58.0%"]

Data Target (decoded): 
 COLT

Model Result (decoded): 
 COLT

Last Char of the Target: T
Last Char of the  Model: T

This pass: Succeeded

Loss:  0.1457354873418808


The **main logic is split** into the following components:  

- **`ModelDetails`** → Contains details on how the model is built, including its attention architecture.  
- **`Model`** → The core structure (backbone) of the transformer model.  
- **`Trainer`** → Implements a **simple training loop**.  
- **`config`** → Standard model and training configurations.  

---

## Samples Provided

This repository includes **two main examples**:  
1️⃣ **"Toy" Example** → Demonstrates **1 to 4-letter word completion**.  
2️⃣ **Shakespeare Completion** → A more complex example attempting to complete Shakespearean text.  
