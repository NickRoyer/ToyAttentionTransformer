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
