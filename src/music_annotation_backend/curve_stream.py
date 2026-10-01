"""Bounded NDJSON stream: cancellation stops between inference batches."""
import asyncio
import json
from threading import Event

from .interactive import relevance_curve


async def stream_curve(asset, request, providers, cache_root):
    queue = asyncio.Queue(maxsize=4)
    loop, stopped = asyncio.get_running_loop(), Event()
    def publish(value):
        if stopped.is_set():
            raise RuntimeError("Relevance analysis cancelled")
        future = asyncio.run_coroutine_threadsafe(queue.put(value), loop)
        while not future.done():
            if stopped.wait(.1):
                future.cancel()
                raise RuntimeError("Relevance analysis cancelled")
        future.result()
    def run():
        try:
            result = relevance_curve(asset, request, providers, cache_root,
                lambda offset, result, total: publish({"type": "chunk", "offset": offset, "total": total, "result": result.model_dump(mode="json", by_alias=True)}), stopped.is_set)
            publish({"type": "complete", "result": result.model_dump(mode="json", by_alias=True)})
        except Exception as error:
            if not stopped.is_set():
                publish({"type": "error", "message": str(error)})
    task = asyncio.create_task(asyncio.to_thread(run))
    try:
        yield json.dumps({"type": "started"}) + "\n"
        while True:
            event = await queue.get()
            yield json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"
            if event["type"] in {"complete", "error"}:
                break
    finally:
        stopped.set()
        # The thread exits after the current native inference call; do not keep
        # the disconnected HTTP request waiting for that call.
        task.add_done_callback(lambda finished: finished.exception() if not finished.cancelled() else None)
