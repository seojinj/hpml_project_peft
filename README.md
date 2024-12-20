# Fine-tuning with QLoRA and Flash Attention 2 on the Yelp Reviews dataset
**Author**: Skylar Jung

This notebook demonstrates how to fine-tune a pre-trained Llama-2 7B model using QLoRA and Flash Attention 2 on the Yelp Reviews dataset. It requires an Ampere or Ada GPU with CUDA 12.2 and Python 3.10.

## Project description
Fine-tuning large language models (LLMs) plays an integral role in today's application of AI models, and while it allows a larger model to be adapted for specific use cases with less resources, it still remains resource-intensive for many applications. Throughout this notebook, we examine the integration of Quantized Low-Rank Adaptation (QLoRA) and Flash Attention to address these challenges. By leveraging sub-4-bit quantization with LoRA and flash attention for efficient attention computation, we observed substantial improvements in memory usage and training speed for fine-tuning. 

## Outline of repository
This repo contains a single file called experiment.ipynb. It can be connected to a machine or container with the appropriate settings, and executed in order.

## Dependencies

Install the required dependencies:

```
pip install -U bitsandbytes==0.45.0
pip install transformers==4.47.0
pip install -U peft==0.10.0
pip install -U accelerate
pip install -U trl
pip install datasets==2.21.0
pip install torch torchvision torchaudio
pip install -q -U git+https://github.com/huggingface/accelerate.git
pip install flash-attn --no-build-isolation
```

## Replicating experiment runs

Execute the cells in order as directed by the text in the notebook file. To ensure that all drivers and packages are compatible, the file must be run on ampere or ada gpus with CUDA 12.2.

## Results
By executing the experiments outlined in the notebook, we observe the following experimental results:

![image](https://github.com/user-attachments/assets/fda415d9-90f3-42ba-a9ae-f1a085c348fc)
