# Crynux SDK

Python SDK for interacting with crynux network.

# Installation

## Installation from source code

1. clone this repo

```shell
git clone https://github.com/crynux-ai/crynux-sdk-python.git
```

2. (Optional, recommended) make a new python venv

```shell
python -m venv venv && source venv/bin/activate
```

3. install the sdk

```shell
cd crynux-sdk-python
pip install .
```

# Usage

## initialize and close

First, initialize the crynux instance

```python
from crynux_sdk import Crynux

async def main():
    crynux = Crynux(privkey="0x")
    await crynux.init()
```

You should close the crynux instance when you don't need it any more.

```python
from crynux_sdk import Crynux

async def main():
    crynux = Crynux(privkey="0x")
    await crynux.init()
    try:
        ...
    finally:
        await crynux.close()
```

Or you can use it as a async context manager.

```python
from crynux_sdk import Crynux

async def main():
    crynux = Crynux(privkey="0x")
    await crynux.init()
    async with crynux:
        ...
```

## deposit tokens

```python
from crynux_sdk import Crynux

async def main():
    crynux = Crynux(privkey="0x")
    await crynux.init()
    async with crynux:
        await crynux.deposit(
            "0x", 10, "ether"
        )
```

Arguments for crynux.deposit_tokens:

* address: Address which deposit tokens to
* amount: Tokens need to deposit, 0 means not to deposit eth
* unit: The unit for eth and cnx tokens, default to "ether"

## generate images

```python
from crynux_sdk import Crynux

async def main():
    crynux = Crynux(privkey="0x")
    await crynux.init()
    async with crynux:
        task_id, start_blocknum, imgs = await crynux.generate_images(
            dst_dir=dst_dir,
            task_fee=30,
            prompt=prompt,
            negative_prompt=negative_prompt,
            base_model="runwayml/stable-diffusion-v1-5",
            task_optional_args={
                "task_config": {
                    "num_images": 1,
                    "safety_checker": False,
                    "seed": 42,
                }
            },
            timeout=360,
        )
```

Arguments for crynux.generate_images:

* dst_dir: Where to store the generated images, should be a string or a pathlib.Path.
            The dst_dir should be existed.
            Generated images will be save in path dst_dir/0.png, dst_dir/1.png and so on.

* task_fee: The cnx tokens you paid for image generation, should be a int.
            You account must have enough cnx tokens before you call this method, 
            or it will failed.

* prompt: The prompt for image generation.
* vram_limit: The GPU VRAM limit for image generation. Crynux network will select nodes 
            with vram larger than vram_limit to generate image for you.
            If vram_limit is None, then the sdk will predict it by the base model.

* base_model: The base model used for image generation, default to runwayml/stable-diffusion-v1-5.
* negative_prompt: The negative prompt for image generation.
* task_optional_args: Optional arguments for image generation. See crynux_sdk.models.sd_args.TaskOptionalArgs for details.
* task_fee_unit: The unit for task fee, default to "ether".
* max_retries: Max retry counts when face network issues, default to 5 times.
* max_timeout_retries: Max retry counts when cannot result images after timeout, default to 3 times.
* timeout: The timeout for image generation in seconds. Default to None, means no timeout.
* wait_interval: The interval in seconds for checking crynux contracts events. Default to 1 second.
* auto_cancel: Whether to cancel the timeout image generation task automatically. Default to True.

* returns: a tuple of task id, blocknum when the task starts, and the result image paths

## finetune LoRA for stable diffusion model

```python
from crynux_sdk import Crynux

async def main():
    crynux = Crynux(privkey="0x")
    await crynux.init()
    async with crynux:
        task_ids, start_blocknums = await crynux.finetune_sd_lora(
            dst_dir,
            task_fee=10,
            gpu_name="NVIDIA GeForce RTX 4090",
            gpu_vram=24,
            model_name="runwayml/stable-diffusion-v1-5",
            dataset_name="lambdalabs/naruto-blip-captions",
            validation_num_images=4,
            learning_rate=1e-4,
            batch_size=1,
            num_train_steps=50,
            max_train_steps=100,
            lr_scheduler="cosine",
            lr_warmup_steps=0,
            rank=4,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            mixed_precision="fp16",
            center_crop=True,
            random_flip=True,
            seed=1337,
        )
```

Arguments for crynux.finetune_sd_lora:

result_checkpoint_path: Should be a string or a pathlib.Path. The directory where the result checkpoint files are stored. 
                        The result lora weight file is `pytorch_lora_weights.safetensors`.
task_fee: The cnx tokens you paid for each finetune task, should be an int.
          You account must have enough cnx tokens before you call this method,
          or it will failed.
gpu_name: The specified GPU name to run this finetune task, should be a string.
gpu_vram: The specified GPU VRAM size to run this finetune task, should be an int, in unit GB.
model_name: The pretrained stable diffusion model to finetune, should be a model identifier from huggingface.co/models.
dataset_name: The name of the Dataset (from the HuggingFace hub) to train on, should be a string.
model_variant: Variant of the model files of the pretrained model identifier from huggingface.co/models, 
               default is None, means no variant.
model_revision: Revision of pretrained model identifier from huggingface.co/models, default is main.
dataset_config_name: The config of the Dataset, default is None means there's only one config.
dataset_image_column: The column of the dataset containing an image, should be a string, default is 'image'.
dataset_caption_column: The column of the dataset containing a caption or a list of captions, should be a string, 
                        default is 'text'.
validation_prompt: A prompt that is used for validation inference during training, should be a string or None.
                   Default is None, means the the prompt is sampled from dataset.
validation_num_images: Number of images that should be generated during validation, should be an int, in range [1, 10].
center_crop: Whether to center crop the input images to the resolution. If not set, the images will be randomly cropped.
             Default is false.
random_flip: Whether to randomly flip images horizontally. Default is false.
rank: Lora attention dimension, should be an int, default is 8.
init_lora_weights: How to initialize the weights of the LoRA layers.Passing True (default) results in the default 
                   initialization from the reference implementation from Microsoft. Passing 'gaussian' results 
                   in Gaussian initialization scaled by the LoRA rank for linear and layers. Setting the initialization
                   to False leads to completely random initialization and is discouraged. 
                   Pass 'loftq' to use LoftQ initialization.
target_modules: List of module names or regex expression of the module names to replace with Lora.
learning_rate: Initial learning rate to use. Default is 1e-4.
batch_size: Batch size for the training dataloader. Default is 16.
gradient_accumulation_steps: Number of updates steps to accumulate before performing a backward/update pass. Default is 1.
prediction_type: The prediction_type that shall be used for training. 
                 Choose between 'epsilon' or 'v_prediction' or leave `None`. 
                 If left to `None` the default prediction type of the scheduler: 
                 `noise_scheduler.config.prediction_type` is chosen. Default is None.
max_grad_norm: Max gradient norm. Default is 1.0.
num_train_epochs: Number of training epochs to perform in one task. Default is 1.
num_train_steps: Number of training steps to perform in one task. Should be an int or None. 
                 Default is None. If not None, overrides 'num_train_epochs'.
max_train_epochs: Total number of training epochs to perform. Default is 1.
max_train_steps: Total number of training steps to perform. Should be an int or None. 
                 Default is None. If not None, overrides 'max_train_epochs'.
scale_lr: Whether to scale the learning rate by the number of gradient accumulation steps, and batch size. Default is true.
resolution: The resolution for input images, all the images in the train/validation dataset will be resized to this resolution.
            Default is 512.
noise_offset: The scale of noise offset. Default is 0.
snr_gamma: SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. Default is None, 
           means to disable rebalancing the loss.
lr_scheduler: The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",
              "constant", "constant_with_warmup"]. Default is "constant".
lr_warmup_steps: Number of steps for the warmup in the lr scheduler. Default is 500.
adam_beta1: The beta1 parameter for the Adam optimizer. Default is 0.9.
adam_beta2: The beta2 parameter for the Adam optimizer. Default is 0.999.
adam_weight_decay: Weight decay to use. Default is 1e-2.
adam_epsilon: Epsilon value for the Adam optimizer. Default is 1e-8.
dataloader_num_workers: Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
                        Default is 0.
mixed_precision: Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). 
                 Default is 'no', means disable mixed precision.
seed: A seed for reproducible training. Default is 0.
input_checkpoint_path: Whether training should be resumed from a previous checkpoint. Should be a path of the previous checkpoint.
                       Default is None, means no previous checkpoint.
task_fee_unit: The unit for task fee, default to "ether".
max_retries: Max retry counts when face network issues, default to 5 times.
max_timeout_retries: Max retry counts when cannot result images after timeout, default to 3 times.
timeout: The timeout for image generation in seconds. Default to None, means no timeout.
wait_interval: The interval in seconds for checking crynux contracts events. Default to 1 second.
auto_cancel: Whether to cancel the timeout image generation task automatically. Default to True.