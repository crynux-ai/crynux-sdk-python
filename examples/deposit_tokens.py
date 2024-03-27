import asyncio
import logging
import os

from crynux_sdk import Crynux


async def main():
    privkey = os.getenv("CRYNUX_PRIVKEY")
    assert (
        privkey is not None
    ), "private key is empty, please provider the private key by set env CRYNUX_PRIVKEY"
    crynux = Crynux(
        privkey=privkey,
    )

    await crynux.init()

    address = "0x906C8eB781cA57F4C223bC2a64028c418060C519"

    async with crynux:
        await crynux.deposit(address, 10, 100)



if __name__ == "__main__":
    logging.basicConfig(
        format="[{asctime}] [{levelname:<8}] {name}: {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{",
        level="INFO",
    )
    asyncio.run(main())
