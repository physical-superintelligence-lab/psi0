"""Multi-client batching server for psi0-simple.

Wire-compatible with `serve_psi0_simple.py` (same `/act` request & response format), but
instead of serving one request at a time it collects requests from several clients and runs
them through the model as a batch:

    POST /act (async)  ->  asyncio.Queue[(request, Future)]
                                  |
                       batcher coroutine: take the first request, then keep
                       accepting while requests keep arriving; stop after
                       `quiet_ms` of silence (or `max_batch` / `max_wait_ms`)
                                  |
                       run_in_executor(1 thread) -> one GPU batch
                                  |
                       resolve each client's Future

Requests that arrive while the GPU is busy simply pile up in the queue and form the next
batch. Only one batch is ever on the GPU.

Two things the batching has to respect:

  * `Psi0Model.predict_action` right-pads mixed-length token sequences like training does, so
    different instructions *can* share a GPU call. We still split a collected batch into
    groups keyed by (instruction, image keys) by default, because padding shifts the obs
    token's positional index and thereby perturbs the action -- see `group_by_instruction`.
    Same-task clients land in a single group and batch for real either way.

  * RTC state (`previous_action`) is per-client, keyed by an id the client sends in
    `history["client_id"]` or `history["session_id"]` (falling back to the peer host).
"""

import asyncio
import json
import sys
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import tyro
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision.transforms import v2

from psi.config.config import ServerConfig
from psi.deploy.helpers import RequestMessage, ResponseMessage
from psi.deploy.serve_psi0_simple import Server
from psi.utils import pad_to_len
from psi.utils.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)


class ServerMultiConfig(ServerConfig):
    """ServerConfig + the knobs that control how requests are grouped into a batch."""

    max_batch: int = 8          # hard cap on samples per GPU call
    quiet_ms: float = 5.0       # stop collecting after this long with no new request
    max_wait_ms: float = 20.0   # never hold the first request of a batch longer than this
    client_ttl_s: float = 300.0 # drop per-client RTC state after this much idle time
    # Psi0Model now right-pads mixed-length instructions exactly like training, so batching
    # them is *possible*. It is off by default because padding shifts the obs token's
    # positional index (it is appended after the padded VLM tokens), which perturbs the action
    # by ~13% of its scale at 24 pad tokens -- i.e. a client's result would depend on the
    # longest instruction it happens to be batched with. Grouping keeps results reproducible.
    group_by_instruction: bool = True


@dataclass
class RtcState:
    previous_action: np.ndarray | None = None
    last_serve_time: float = 0.0


@dataclass
class Pending:
    """One in-flight client request as it moves through the batcher."""

    payload: Dict[str, Any]
    peer: str
    future: "asyncio.Future[Any]"
    enqueued_at: float

    # filled in by _preprocess
    client_id: str = ""
    instruction: str = ""
    images: List[Any] = field(default_factory=list)   # transformed PIL/tensor images
    state: torch.Tensor | None = None                 # (1, To, Ds)
    reset: bool = False
    error: BaseException | None = None

    group_by_instruction: bool = True

    @property
    def group_key(self) -> Tuple[str, Tuple[str, ...]]:
        instruction = self.instruction if self.group_by_instruction else ""
        return (instruction, tuple(self.payload["image"].keys()))


class BatchServer(Server):
    """`Server` with a batching front-end. Model loading / config parsing are inherited."""

    def __init__(
        self,
        *args,
        max_batch: int = 8,
        quiet_ms: float = 5.0,
        max_wait_ms: float = 20.0,
        client_ttl_s: float = 300.0,
        group_by_instruction: bool = True,
        **kwargs,
    ):
        # The batching path only implements the train-time (frozen-prefix) RTC sampler,
        # and _use_rtc/_infer_rtc are shared by the whole batch. Base `Server` resolves
        # rtc_mode="auto" to "test_time" on a --model.no-rtc checkpoint, which this
        # subclass would then silently serve through the train-time sampler.
        kwargs.setdefault("rtc_mode", "train")
        super().__init__(*args, **kwargs)
        if self.rtc_mode == "test_time":
            raise NotImplementedError(
                "serve_psi0_simple_multi implements train-time RTC only; serve this "
                "checkpoint with serve_psi0_simple (single client) for test-time RTC.")

        assert max_batch >= 1, "max_batch must be >= 1"
        assert max_wait_ms >= quiet_ms, "max_wait_ms must be >= quiet_ms"
        self.max_batch = max_batch
        self.quiet_s = quiet_ms / 1000.0
        self.max_wait_s = max_wait_ms / 1000.0
        self.client_ttl_s = client_ttl_s
        self.group_by_instruction = group_by_instruction

        # image transform is stateless -> build once instead of per request
        self._img_transform = v2.Compose(
            [self.model_transform.resize(), self.model_transform.center_crop()]
        )

        # per-client RTC state; only ever touched from the single inference thread
        self.rtc_states: "OrderedDict[str, RtcState]" = OrderedDict()
        self._warned_no_client_id: set[str] = set()

        self._queue: "asyncio.Queue[Pending]" = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="psi0-infer")
        self._stats = deque(maxlen=50)  # (batch_size, num_groups, latency_s)
        self._total_requests = 0
        self._total_batches = 0

    # ------------------------------------------------------------------ per-request

    def _client_id(self, req: Pending, history: Dict[str, Any]) -> str:
        history = history if isinstance(history, dict) else {}
        # `session_id` is what the SIMPLE eval client sends (simple/baselines/psi0.py).
        cid = history.get("client_id") or history.get("session_id")
        if cid:
            return str(cid)
        # No id from the client: fall back to the peer host. Not the port -- a client that
        # posts without keep-alive (requests.post per step) draws a fresh ephemeral port
        # every call, which would key a brand new RTC state on every request.
        if req.peer not in self._warned_no_client_id:
            self._warned_no_client_id.add(req.peer)
            overwatch.warning(
                f"client {req.peer} sent no history['client_id']; falling back to the peer "
                f"address for RTC state. Reconnects will reset that client's RTC history."
            )
        return req.peer

    def _preprocess(self, req: Pending) -> None:
        """Decode one payload into model-ready tensors. Failures are attached to the request
        so a single bad payload cannot take the batch down."""
        try:
            request = RequestMessage.deserialize(req.payload)
            history = request.history if isinstance(request.history, dict) else {}

            req.client_id = self._client_id(req, history)
            req.instruction = request.instruction
            req.reset = "reset" in history
            req.images = [
                self._img_transform(Image.fromarray(img)) for img in request.image.values()
            ]

            states = torch.from_numpy(request.state["states"].copy())
            if self.maxmin.normalize_state:  # type: ignore
                states = torch.as_tensor(
                    self.maxmin.normalize_state_func(
                        pad_to_len(states.numpy(), self.pad_state_dim, dim=1)[0]
                    )
                )
            req.state = states.unsqueeze(0).to(self.device)  # (1, To, Ds)
        except BaseException as e:  # noqa: BLE001 - reported back to this one client
            import traceback

            overwatch.warning(f"preprocess failed for {req.peer}:\n{traceback.format_exc()}")
            req.error = e

    # ------------------------------------------------------------------ inference

    def _stack(self, group: List[Pending]) -> Tuple[List[List[Any]], torch.Tensor, List[str]]:
        observations = [r.images for r in group]
        states = torch.cat([r.state for r in group], dim=0)  # type: ignore[arg-type]
        instructions = [r.instruction for r in group]
        return observations, states, instructions

    def _infer_plain(self, group: List[Pending]) -> torch.Tensor:
        observations, states, instructions = self._stack(group)
        return self.model.predict_action(
            observations=observations,
            states=states,
            instructions=instructions,
            num_inference_steps=self.num_inference_steps,
            traj2ds=None,
        )

    def _infer_rtc(self, group: List[Pending]) -> torch.Tensor:
        observations, states, instructions = self._stack(group)
        # Shift each client's own previous chunk by exec_shift, the rows it executed.
        prev = np.stack(
            [
                np.concatenate(
                    [
                        self.rtc_states[r.client_id].previous_action[self.exec_shift:, :],  # type: ignore[index]
                        np.zeros((self.exec_shift, self.Da), dtype=np.float32),
                    ],
                    axis=0,
                )
                for r in group
            ],
            axis=0,
        )  # (B, Tp, Da)
        return self.model.predict_action_with_training_rtc_flow(
            observations=observations,
            states=states,
            instructions=instructions,
            num_inference_steps=self.num_inference_steps,
            traj2ds=None,
            prev_actions=torch.from_numpy(prev).to(self.device),
            inference_delay=self.rtc_inference_delay,
            max_delay=self.rtc_max_delay,
        )

    def _use_rtc(self, req: Pending) -> bool:
        if not self.enable_rtc:
            return False
        state = self.rtc_states.get(req.client_id)
        return state is not None and state.previous_action is not None and not req.reset

    def _finish(self, req: Pending, raw: np.ndarray) -> Dict[str, Any]:
        """raw: (Tp, Da) normalized prediction for this one client."""
        raw = raw.reshape(-1, self.Da)
        pred_actions = self.maxmin.denormalize(raw)[: self.Ta]
        state = self.rtc_states.setdefault(req.client_id, RtcState())
        state.previous_action = raw.copy().astype(np.float32)
        state.last_serve_time = time.monotonic()
        self.rtc_states.move_to_end(req.client_id)
        return ResponseMessage(pred_actions, 0.0).serialize()  # type: ignore[arg-type]

    def _run_subbatch(self, sub: List[Pending], rtc: bool, results: Dict[int, Any]) -> None:
        """Run one homogeneous sub-batch; on failure retry its members one at a time so a
        single poisoned request cannot fail everyone else's."""
        fn = self._infer_rtc if rtc else self._infer_plain
        try:
            out = fn(sub).cpu().numpy()
            for i, req in enumerate(sub):
                results[id(req)] = self._finish(req, out[i])
            return
        except BaseException as e:  # noqa: BLE001
            import traceback

            if len(sub) == 1:
                overwatch.warning(f"inference failed for {sub[0].client_id}:\n{traceback.format_exc()}")
                results[id(sub[0])] = {"status": str(e)}
                return
            overwatch.warning(
                f"batched inference (B={len(sub)}, rtc={rtc}) failed, retrying one by one:\n"
                f"{traceback.format_exc()}"
            )

        for req in sub:
            self._run_subbatch([req], rtc, results)

    def _run_batch(self, batch: List[Pending]) -> List[Tuple[Pending, Any]]:
        """Runs on the single inference thread. Returns (request, serialized-content) pairs."""
        self._evict_stale_clients()
        results: Dict[int, Any] = {}

        for req in batch:
            self._preprocess(req)
            if req.error is not None:
                results[id(req)] = {"status": str(req.error)}

        live = [r for r in batch if r.error is None]

        groups: "OrderedDict[Tuple[str, Tuple[str, ...]], List[Pending]]" = OrderedDict()
        for req in live:
            groups.setdefault(req.group_key, []).append(req)

        if len(groups) > 1:
            overwatch.info(
                f"batch of {len(live)} split into {len(groups)} groups by (instruction, image keys): "
                f"{[ (k[0][:40], len(v)) for k, v in groups.items() ]}"
            )

        for members in groups.values():
            # a group can still mix RTC-conditioned and fresh clients (first step / reset)
            for chunk_start in range(0, len(members), self.max_batch):
                chunk = members[chunk_start : chunk_start + self.max_batch]
                rtc_reqs = [r for r in chunk if self._use_rtc(r)]
                plain_reqs = [r for r in chunk if not self._use_rtc(r)]
                if plain_reqs:
                    self._run_subbatch(plain_reqs, rtc=False, results=results)
                if rtc_reqs:
                    self._run_subbatch(rtc_reqs, rtc=True, results=results)

        return [(req, results[id(req)]) for req in batch]

    def _evict_stale_clients(self) -> None:
        cutoff = time.monotonic() - self.client_ttl_s
        while self.rtc_states:
            cid, state = next(iter(self.rtc_states.items()))
            if state.last_serve_time >= cutoff:
                break
            self.rtc_states.pop(cid)
            overwatch.info(f"evicted idle client state: {cid}")

    # ------------------------------------------------------------------ batching loop

    async def _collect(self) -> List[Pending]:
        """The batching rule: block for the first request, then keep taking requests until
        `quiet_ms` passes with none arriving (or the batch/wait caps are hit)."""
        first = await self._queue.get()
        batch = [first]
        deadline = first.enqueued_at + self.max_wait_s

        while len(batch) < self.max_batch:
            timeout = min(self.quiet_s, deadline - time.monotonic())
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), timeout))
            except asyncio.TimeoutError:
                break  # quiet period elapsed -> go run
        return batch

    async def _batcher(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch: List[Pending] = []
            try:
                batch = await self._collect()
                started = time.monotonic()
                pairs = await loop.run_in_executor(self._executor, self._run_batch, batch)
                latency = time.monotonic() - started

                for req, content in pairs:
                    if not req.future.done():
                        req.future.set_result(content)

                self._total_requests += len(batch)
                self._total_batches += 1
                n_groups = len({r.group_key for r in batch if r.error is None})
                self._stats.append((len(batch), n_groups, latency))
                overwatch.info(
                    f"batch B={len(batch)} groups={n_groups} took {latency*1000:.0f}ms "
                    f"(queue depth {self._queue.qsize()})"
                )
            except asyncio.CancelledError:
                raise
            except BaseException:  # noqa: BLE001 - the loop must never die
                import traceback

                overwatch.warning(f"batcher loop error:\n{traceback.format_exc()}")
                for req in batch:
                    if not req.future.done():
                        req.future.set_result({"status": "server batch error"})

    # ------------------------------------------------------------------ endpoints

    async def predict_action(self, payload: Dict[str, Any], request: Request) -> JSONResponse:  # type: ignore[override]
        peer = request.client.host if request.client else "unknown"
        pending = Pending(
            payload=payload,
            peer=peer,
            future=asyncio.get_running_loop().create_future(),
            enqueued_at=time.monotonic(),
            group_by_instruction=self.group_by_instruction,
        )
        await self._queue.put(pending)
        return JSONResponse(content=await pending.future)

    def info(self) -> JSONResponse:  # type: ignore[override]
        content = json.loads(super().info().body)
        content["batching"] = {
            "max_batch": self.max_batch,
            "quiet_ms": self.quiet_s * 1000.0,
            "max_wait_ms": self.max_wait_s * 1000.0,
            "client_ttl_s": self.client_ttl_s,
            "group_by_instruction": self.group_by_instruction,
        }
        content["expected_keys"]["history"] = {
            "reset": "optional",
            "client_id": "optional but recommended - keys per-client RTC state; "
                         "`session_id` is accepted too, else the peer host",
        }
        return JSONResponse(content=content)

    def health(self) -> JSONResponse:
        sizes = [s[0] for s in self._stats]
        lats = [s[2] for s in self._stats]
        return JSONResponse(
            content={
                "status": "ok",
                "queue_depth": self._queue.qsize(),
                "total_requests": self._total_requests,
                "total_batches": self._total_batches,
                "recent_avg_batch_size": float(np.mean(sizes)) if sizes else 0.0,
                "recent_max_batch_size": int(max(sizes)) if sizes else 0,
                "recent_avg_batch_latency_ms": float(np.mean(lats) * 1000.0) if lats else 0.0,
                "active_clients": len(self.rtc_states),
            }
        )

    def build_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            task = asyncio.create_task(self._batcher())
            overwatch.info(
                f"batcher started (max_batch={self.max_batch}, quiet={self.quiet_s*1000:.0f}ms, "
                f"max_wait={self.max_wait_s*1000:.0f}ms)"
            )
            try:
                yield
            finally:
                task.cancel()
                self._executor.shutdown(wait=False)

        app = FastAPI(lifespan=lifespan)
        app.post("/act")(self.predict_action)
        app.get("/info")(self.info)
        app.get("/health")(self.health)
        return app

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:  # type: ignore[override]
        self.app = self.build_app()
        overwatch.info(f"Server listens on {host}:{port}")
        try:
            uvicorn.run(self.app, host=host, port=port)
        except Exception as e:
            overwatch.warning(f"Server crashed, {e}")
        finally:
            overwatch.info("Server stopped.")
            exit(1)


def serve(cfg: ServerMultiConfig) -> None:
    overwatch.info("Server :: Initializing Psi0 (multi-client batching)")
    assert cfg.policy is not None, "which policy to serve?"

    # asyncio.Queue binds to the running loop lazily (py>=3.10), so constructing the server
    # before uvicorn starts its loop is safe.
    server = BatchServer(
        cfg.policy,
        Path(cfg.run_dir),
        cfg.ckpt_step,
        cfg.device,
        cfg.rtc,
        cfg.action_exec_horizon,
        rtc_inference_delay=cfg.rtc_inference_delay,
        min_exec_horizon=cfg.min_exec_horizon,
        num_inference_steps=cfg.num_inference_steps,
        max_batch=cfg.max_batch,
        quiet_ms=cfg.quiet_ms,
        max_wait_ms=cfg.max_wait_ms,
        client_ttl_s=cfg.client_ttl_s,
    )
    overwatch.info("Server :: Spinning Up")
    server.run(cfg.host, cfg.port)


def main():
    overwatch.info("Start Serving from uv")
    overwatch.info(f"Args: {sys.argv}")
    from dotenv import load_dotenv

    assert load_dotenv()
    config = tyro.cli(
        ServerMultiConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=sys.argv[1:]
    )
    serve(config)


if __name__ == "__main__":
    from dotenv import load_dotenv

    assert load_dotenv()
    config = tyro.cli(ServerMultiConfig, config=(tyro.conf.ConsolidateSubcommandArgs,))
    serve(config)
