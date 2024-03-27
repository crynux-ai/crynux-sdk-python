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
import Crynux from crynux_sdk

async def main():
    crynux = Crynux(privkey="0x")
    await crynux.init()
```

You should close the crynux instance when you don't need it any more.

```python
import Crynux from crynux_sdk

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
import Crynux from crynux_sdk

async def main():
    crynux = Crynux(privkey="0x")
    await crynux.init()
    async with crynux:
        ...
```

## deposit tokens

```python
import Crynux from crynux_sdk

async def main():
    crynux = Crynux(privkey="0x")
    await crynux.init()
    async with crynux:
        await crynux.deposit_tokens(
            "0x", 10, 10, "ether"
        )
```

Arguments for crynux.deposit_tokens:

* address: Address which deposit tokens to
* eth: Eth tokens need to deposit, 0 means not to deposit eth
* cnx: Cnx tokens need to deposit, 0 means not to deposit cnx
* unit: The unit for eth and cnx tokens, default to "ether"

## generate images

```python
import Crynux from crynux_sdk

async def main():
    crynux = Crynux(privkey="0x")
    await crynux.init()
    async with crynux:
        await crynux.generate_images(
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

