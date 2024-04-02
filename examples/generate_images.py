import asyncio
import logging
import os
import pathlib

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

    prompt = (
        "best quality, ultra high res, photorealistic++++, 1girl, off-shoulder sweater, smiling, "
        "faded ash gray messy bun hair+, border light, depth of field, looking at "
        "viewer, closeup"
    )

    negative_prompt = (
        "paintings, sketches, worst quality+++++, low quality+++++, normal quality+++++, lowres, "
        "normal quality, monochrome++, grayscale++, skin spots, acnes, skin blemishes, "
        "age spot, glans"
    )

    await crynux.init()
    async with crynux:
        task_id, result_imgs = await crynux.generate_images(
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

    if all(img.exists() for img in result_imgs):
        print(f"generate image successfully in task {task_id}")
    else:
        print("missing result image")


if __name__ == "__main__":
    logging.basicConfig(
        format="[{asctime}] [{levelname:<8}] {name}: {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{",
        level="INFO",
    )
    asyncio.run(main())
