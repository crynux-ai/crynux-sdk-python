import asyncio
import logging
import os
import pathlib

from web3 import Web3

from crynux_sdk import Crynux


async def main():
    privkey = os.getenv("CRYNUX_PRIVKEY")
    assert (
        privkey is not None
    ), "private key is empty, please provider the private key by set env CRYNUX_PRIVKEY"
    crynux = Crynux(
        privkey=privkey,
    )

    dst_dir = pathlib.Path("./data")
    if not dst_dir.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)


    await crynux.init()
    async with crynux:
        await crynux.finetune_sd_lora(
            dst_dir,
            task_fee=10,
            gpu_name="NVIDIA GeForce GTX 1070 Ti",
            gpu_vram=8,
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

    if os.path.exists(dst_dir / "checkpoint"):
        print(f"finetune lora successfully")
    else:
        print("finetune lora failed")


if __name__ == "__main__":
    logging.basicConfig(
        format="[{asctime}] [{levelname:<8}] {name}: {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{",
        level="INFO",
    )
    asyncio.run(main())
