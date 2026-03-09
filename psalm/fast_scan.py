from __future__ import annotations

import contextlib
import io
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from queue import Empty
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

import torch
from tqdm.auto import tqdm

from psalm.inference.decoder import annotate_domains
from psalm.psalm_model import (
    _DecodedDomains,
    _PreparedDecodeInputs,
    _load_fasta_records_checked,
    _normalize_sequence_text,
)
from psalm.report import DomainTuple, HitRow

if TYPE_CHECKING:
    from psalm.psalm_model import PSALM


DEFAULT_BUNDLE_SIZE = 8
DEFAULT_MAX_QUEUED_SEQS_PER_WORKER = 10
WORKER_READY_TIMEOUT_S = 60.0
WORKER_SHUTDOWN_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class _FastRecord:
    order_idx: int
    seq_id: str
    sequence: str


@dataclass(frozen=True)
class _FastBatch:
    records: List[_FastRecord]


@dataclass(frozen=True)
class _DecodeTask:
    order_idx: int
    seq_id: str
    sequence: str
    prepared: _PreparedDecodeInputs
    inf_dt: float


@dataclass(frozen=True)
class _DecodeBundle:
    tasks: List[_DecodeTask]
    beam_size: int
    prior_mid_to_start: float
    prior_stop_to_mid: float
    prior_stop_to_start: float
    refine_extended: bool


@dataclass(frozen=True)
class _DecodeResult:
    order_idx: int
    seq_id: str
    domains: List[DomainTuple]
    dec_dt: float
    error: Optional[str] = None


@dataclass(frozen=True)
class _DecodeBundleResult:
    results: List[_DecodeResult]


@dataclass(frozen=True)
class _WorkerReady:
    error: Optional[str] = None


@dataclass
class _CompletedFastScan:
    seq_id: str
    sequence: str
    domains_scored: List[DomainTuple]
    hit_rows: List[HitRow]
    score_pass: int
    total_domains_raw: int
    total_fams: int
    kept_families: int
    inf_dt: float
    dec_dt: float


def _normalize_path(path: Optional[str], default_ext: str) -> Optional[str]:
    if path is None:
        return None
    base, ext = os.path.splitext(path)
    return path if ext else f"{base}{default_ext}"


def _load_fast_records(fasta: str) -> List[_FastRecord]:
    records = _load_fasta_records_checked(fasta)
    used_ids: Dict[str, int] = {}
    unnamed = 0
    normalized: List[_FastRecord] = []
    for order_idx, rec in enumerate(records):
        rec_id = (rec.id or "").strip()
        if rec_id == "":
            unnamed += 1
            rec_id = f"seq{unnamed}"
        if rec_id in used_ids:
            used_ids[rec_id] += 1
            rec_id = f"{rec_id}_{used_ids[rec_id]}"
        else:
            used_ids[rec_id] = 1
        sequence = _normalize_sequence_text(
            str(rec.seq),
            f"sequence '{rec_id}' in FASTA '{fasta}'",
        )
        normalized.append(_FastRecord(order_idx=order_idx, seq_id=rec_id, sequence=sequence))
    return normalized


def _create_batches(records: List[_FastRecord], max_tokens_per_batch: int) -> List[_FastBatch]:
    batches: List[_FastBatch] = []
    i = 0
    total = len(records)
    while i < total:
        batch_records: List[_FastRecord] = []
        current_max = 0
        while i < total:
            record = records[i]
            tokens = len(record.sequence) + 2
            if not batch_records and tokens > max_tokens_per_batch:
                batch_records.append(record)
                i += 1
                break
            proposed_max = max(current_max, tokens)
            if batch_records and (len(batch_records) + 1) * proposed_max > max_tokens_per_batch:
                break
            batch_records.append(record)
            current_max = proposed_max
            i += 1
        batches.append(_FastBatch(records=batch_records))
    return batches


def _build_worker_warmup_bundle(model: "PSALM") -> _DecodeBundle:
    warmup_family = None
    for pfam, info in model.label_mapping.items():
        if pfam != "None" and isinstance(info, dict):
            warmup_family = info
            break
    if warmup_family is None:
        raise RuntimeError("Could not build fast worker warmup task: no family labels available.")

    sequence = "AAAA"
    logits = torch.full((len(sequence), model.classes), -50.0, dtype=torch.float32)
    logits[:, model.none_label] = -4.0
    state_path = [
        int(warmup_family["start"]),
        int(warmup_family["middle"]),
        int(warmup_family["middle"]),
        int(warmup_family["stop"]),
    ]
    for pos, state_idx in enumerate(state_path):
        logits[pos, state_idx] = 0.0
    log_probs = torch.log_softmax(logits, dim=-1)
    prepared = model._prepare_decode_inputs(log_probs)
    if prepared.sub_emits_cpu is None or prepared.none_sub is None:
        raise RuntimeError("Could not build fast worker warmup task.")
    return _DecodeBundle(
        tasks=[_DecodeTask(order_idx=-1, seq_id="__warmup__", sequence=sequence, prepared=prepared, inf_dt=0.0)],
        beam_size=1,
        prior_mid_to_start=3.4e-5,
        prior_stop_to_mid=4.817e-3,
        prior_stop_to_start=5.540e-3,
        refine_extended=False,
    )


def _decode_bundle(bundle: _DecodeBundle, decoder_resources) -> _DecodeBundleResult:
    (
        inverse_label_mapping,
        label_mapping,
        from_list,
        to_list,
        lp_list,
        role_map,
        state_pfam_ids_full,
        state_role_ids_full,
        id_to_pfam,
    ) = decoder_resources
    results: List[_DecodeResult] = []
    for task in bundle.tasks:
        try:
            t0 = time.time()
            _domains_original, domains, _best_path, _collapsed, _scores, _meta = annotate_domains(
                task.prepared.sub_emits_cpu,
                task.prepared.keep_full_cpu,
                task.sequence,
                inverse_label_mapping,
                label_mapping,
                from_list,
                to_list,
                lp_list,
                role_map,
                state_pfam_ids_full,
                state_role_ids_full,
                id_to_pfam,
                task.prepared.none_sub,
                beam_size=bundle.beam_size,
                prior_mid_to_start=bundle.prior_mid_to_start,
                prior_stop_to_mid=bundle.prior_stop_to_mid,
                prior_stop_to_start=bundle.prior_stop_to_start,
                refine_extended=bundle.refine_extended,
            )
            results.append(
                _DecodeResult(
                    order_idx=task.order_idx,
                    seq_id=task.seq_id,
                    domains=domains,
                    dec_dt=time.time() - t0,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive worker path
            results.append(
                _DecodeResult(
                    order_idx=task.order_idx,
                    seq_id=task.seq_id,
                    domains=[],
                    dec_dt=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return _DecodeBundleResult(results=results)


def _decode_worker(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    ready_queue: mp.Queue,
    decoder_resources,
    warmup_bundle: _DecodeBundle,
) -> None:
    try:
        _decode_bundle(warmup_bundle, decoder_resources)
        ready_queue.put(_WorkerReady())
    except Exception as exc:  # pragma: no cover - defensive worker path
        ready_queue.put(_WorkerReady(error=f"{type(exc).__name__}: {exc}"))
        return

    while True:
        bundle = task_queue.get()
        if bundle is None:
            break
        result_queue.put(_decode_bundle(bundle, decoder_resources))


class FastWorkerPoolManager:
    def __init__(self, model: "PSALM") -> None:
        self.ctx = mp.get_context("spawn")
        self.decoder_resources = (
            model.inverse_label_mapping,
            model.label_mapping,
            model.from_list,
            model.to_list,
            model.lp_list,
            model.role_map,
            model.state_pfam_ids_full,
            model.state_role_ids_full,
            model.id_to_pfam,
        )
        self.warmup_bundle = _build_worker_warmup_bundle(model)
        self.task_queue: Optional[mp.Queue] = None
        self.result_queue: Optional[mp.Queue] = None
        self.ready_queue: Optional[mp.Queue] = None
        self._workers: List[mp.Process] = []
        self.target_size = 0

    @property
    def current_size(self) -> int:
        self._prune_dead_workers()
        return len(self._workers)

    def _ensure_queues(self) -> None:
        if self.task_queue is None:
            self.task_queue = self.ctx.Queue()
        if self.result_queue is None:
            self.result_queue = self.ctx.Queue()
        if self.ready_queue is None:
            self.ready_queue = self.ctx.Queue()

    def _spawn_workers(self, count: int) -> None:
        self._ensure_queues()
        for _ in range(count):
            proc = self.ctx.Process(
                target=_decode_worker,
                args=(
                    self.task_queue,
                    self.result_queue,
                    self.ready_queue,
                    self.decoder_resources,
                    self.warmup_bundle,
                ),
            )
            proc.start()
            self._workers.append(proc)

    def _prune_dead_workers(self) -> None:
        alive: List[mp.Process] = []
        for proc in self._workers:
            if proc.is_alive():
                alive.append(proc)
            else:
                proc.join(timeout=0)
        self._workers = alive

    def _wait_until_ready(self, count: int) -> None:
        if count <= 0 or self.ready_queue is None:
            return
        deadline = time.time() + WORKER_READY_TIMEOUT_S
        remaining = count
        while remaining > 0:
            timeout = max(0.1, deadline - time.time())
            if timeout <= 0:
                raise RuntimeError("Timed out while warming fast decode workers.")
            try:
                ready = self.ready_queue.get(timeout=timeout)
            except Empty as exc:
                self._prune_dead_workers()
                raise RuntimeError("Timed out while warming fast decode workers.") from exc
            if ready.error is not None:
                raise RuntimeError(f"Fast decode worker warmup failed: {ready.error}")
            remaining -= 1

    def _shrink_to(self, target_size: int) -> None:
        current = self.current_size
        if target_size >= current:
            return
        if self.task_queue is None:
            self._workers = []
            return
        for _ in range(current - target_size):
            self.task_queue.put(None)
        deadline = time.time() + WORKER_SHUTDOWN_TIMEOUT_S
        while time.time() < deadline:
            self._prune_dead_workers()
            if len(self._workers) <= target_size:
                return
            time.sleep(0.05)
        raise RuntimeError("Timed out while shrinking fast decode worker pool.")

    def ensure_size(self, target_size: int, status_callback: Optional[Callable[[str], None]] = None) -> None:
        if target_size < 0:
            raise ValueError("cpu_workers must be >= 0.")
        current = self.current_size
        if target_size == current:
            self.target_size = target_size
            return
        if target_size < current:
            self._shrink_to(target_size)
            self.target_size = target_size
            return
        delta = target_size - current
        if delta > 0 and status_callback is not None:
            status_callback(f"Warming {delta} CPU decode worker(s)...")
        self._spawn_workers(delta)
        self._wait_until_ready(delta)
        self.target_size = target_size

    def submit_bundle(self, bundle: _DecodeBundle) -> None:
        if self.task_queue is None or self.current_size <= 0:
            raise RuntimeError("Fast decode worker pool is not ready.")
        self.task_queue.put(bundle)

    def get_result(self, timeout: float) -> Optional[_DecodeBundleResult]:
        if self.result_queue is None:
            return None
        try:
            return self.result_queue.get(timeout=timeout)
        except Empty:
            return None

    def has_live_workers(self) -> bool:
        self._prune_dead_workers()
        return bool(self._workers)

    def close(self) -> None:
        current = self.current_size
        if self.task_queue is not None:
            for _ in range(current):
                self.task_queue.put(None)
        deadline = time.time() + WORKER_SHUTDOWN_TIMEOUT_S
        while time.time() < deadline:
            self._prune_dead_workers()
            if not self._workers:
                break
            time.sleep(0.05)
        for proc in self._workers:
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=1.0)
        self._workers = []
        self.target_size = 0


def _render_report(model: "PSALM", result: _CompletedFastScan, *, score_thresh: float, evalue_thresh: float, refine_extended: bool, beam_size: int) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        model._render_scan_output(
            sequence=result.sequence,
            seq_id=result.seq_id,
            hit_rows=result.hit_rows,
            score_thresh=score_thresh,
            evalue_thresh=evalue_thresh,
            score_pass=result.score_pass,
            total_domains_raw=result.total_domains_raw,
            total_fams=result.total_fams,
            kept_families=result.kept_families,
            domains_original_scored=[],
            gamma=None,
            best_path=None,
            verbose=False,
            refine_extended=refine_extended,
            beam_size=beam_size,
            inf_dt=result.inf_dt,
            dec_dt=result.dec_dt,
        )
    return buffer.getvalue().rstrip("\n")


def _default_bundle_size(cpu_workers: int) -> int:
    if cpu_workers <= 0:
        return 1
    return min(16, max(4, cpu_workers * 2))


def _resolve_cpu_workers(
    requested_cpu_workers: Optional[int],
    worker_pool: Optional[FastWorkerPoolManager],
) -> int:
    if requested_cpu_workers is not None:
        return requested_cpu_workers
    if worker_pool is not None:
        return worker_pool.target_size
    return 0


def scan_fasta_fast(
    model: "PSALM",
    *,
    fasta: str,
    score_thresh: float,
    beam_size: int,
    dataset_size: Optional[float],
    evalue_thresh: float,
    prior_mid_to_start: float,
    prior_stop_to_mid: float,
    prior_stop_to_start: float,
    refine_extended: bool,
    to_tsv: Optional[str],
    to_txt: Optional[str],
    sort: bool,
    cpu_workers: Optional[int],
    max_tokens_per_batch: int,
    max_queued_seqs: Optional[int] = None,
    _print_output: bool = True,
    _worker_pool_manager: Optional[FastWorkerPoolManager] = None,
    _status_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, List[DomainTuple]]:
    if dataset_size is not None and dataset_size <= 0:
        raise ValueError("dataset_size must be > 0 when provided.")
    if cpu_workers is not None and cpu_workers < 0:
        raise ValueError("cpu_workers must be >= 0.")
    if max_tokens_per_batch <= 0:
        raise ValueError("max_tokens_per_batch must be > 0.")
    if max_queued_seqs is not None and max_queued_seqs <= 0:
        raise ValueError("max_queued_seqs must be > 0 when provided.")

    emit_console = bool(_print_output)
    status_callback = _status_callback
    if status_callback is None:
        status_callback = lambda msg: print(msg, flush=True)

    model._ensure_scan_ready()
    records = _load_fast_records(fasta)
    if sort:
        records = sorted(records, key=lambda record: len(record.sequence), reverse=True)
    records = [
        _FastRecord(order_idx=idx, seq_id=record.seq_id, sequence=record.sequence)
        for idx, record in enumerate(records)
    ]
    batches = _create_batches(records, max_tokens_per_batch=max_tokens_per_batch)
    effective_dataset_size = float(dataset_size) if dataset_size is not None else float(max(1, len(records)))

    txt_path = _normalize_path(to_txt, ".txt")
    emit_text_file = bool(txt_path)
    completed: List[Optional[_CompletedFastScan]] = [None] * len(records)

    resolved_cpu_workers = _resolve_cpu_workers(cpu_workers, _worker_pool_manager)
    bundle_size = _default_bundle_size(resolved_cpu_workers)
    effective_max_queued_seqs = (
        max_queued_seqs
        if max_queued_seqs is not None
        else max(bundle_size, DEFAULT_MAX_QUEUED_SEQS_PER_WORKER * max(1, resolved_cpu_workers))
    )

    local_worker_pool = None
    active_worker_pool = _worker_pool_manager
    if resolved_cpu_workers > 0:
        if active_worker_pool is None:
            local_worker_pool = FastWorkerPoolManager(model)
            active_worker_pool = local_worker_pool
        active_worker_pool.ensure_size(resolved_cpu_workers, status_callback=status_callback)
    elif active_worker_pool is not None and cpu_workers is not None:
        active_worker_pool.ensure_size(0, status_callback=status_callback)

    pbar = tqdm(
        total=len(records),
        desc="Fast scan",
        unit="seq",
        disable=(not emit_console) or (not sys.stderr.isatty()),
    )
    pending_tasks: Dict[int, _DecodeTask] = {}
    bundle_buffer: List[_DecodeTask] = []

    def _store_completed(
        order_idx: int,
        seq_id: str,
        sequence: str,
        prepared: _PreparedDecodeInputs,
        decoded: _DecodedDomains,
        inf_dt: float,
    ) -> None:
        finalized = model._finalize_scan_result(
            seq_id=seq_id,
            decoded=decoded,
            dataset_size=effective_dataset_size,
            score_thresh=score_thresh,
            evalue_thresh=evalue_thresh,
            include_verbose_artifacts=False,
        )
        completed[order_idx] = _CompletedFastScan(
            seq_id=seq_id,
            sequence=sequence,
            domains_scored=finalized.domains_scored,
            hit_rows=finalized.hit_rows,
            score_pass=finalized.score_pass,
            total_domains_raw=finalized.total_domains_raw,
            total_fams=prepared.total_fams,
            kept_families=prepared.kept_families,
            inf_dt=inf_dt,
            dec_dt=decoded.dec_dt,
        )
        pbar.update(1)

    def _flush_bundle(force: bool = False) -> None:
        if active_worker_pool is None or not bundle_buffer:
            return
        if force or len(bundle_buffer) >= bundle_size:
            active_worker_pool.submit_bundle(
                _DecodeBundle(
                    tasks=list(bundle_buffer),
                    beam_size=beam_size,
                    prior_mid_to_start=prior_mid_to_start,
                    prior_stop_to_mid=prior_stop_to_mid,
                    prior_stop_to_start=prior_stop_to_start,
                    refine_extended=refine_extended,
                )
            )
            bundle_buffer.clear()

    def _drain_results(block: bool) -> None:
        if active_worker_pool is None:
            return
        received_any = False
        while pending_tasks:
            timeout = 0.1 if block and not received_any else 0.0
            result_bundle = active_worker_pool.get_result(timeout=timeout)
            if result_bundle is None:
                if not block or received_any:
                    break
                if not active_worker_pool.has_live_workers():
                    raise RuntimeError("Fast scan decode worker exited unexpectedly.")
                continue
            received_any = True
            for decode_result in result_bundle.results:
                task = pending_tasks.pop(decode_result.order_idx, None)
                if task is None:
                    continue
                if decode_result.error is not None:
                    raise RuntimeError(
                        f"Fast scan decode failed for sequence '{decode_result.seq_id}': {decode_result.error}"
                    )
                decoded = _DecodedDomains(
                    domains=decode_result.domains,
                    domains_original=[],
                    gamma=None,
                    best_path=None,
                    dec_dt=decode_result.dec_dt,
                )
                _store_completed(
                    order_idx=task.order_idx,
                    seq_id=task.seq_id,
                    sequence=task.sequence,
                    prepared=task.prepared,
                    decoded=decoded,
                    inf_dt=task.inf_dt,
                )

    try:
        for batch in batches:
            batch_sequences = [record.sequence for record in batch.records]
            t0 = time.time()
            batch_log_probs = model.batch_infer(batch_sequences, max_batch_size=len(batch_sequences))
            batch_inf_dt = time.time() - t0
            per_sequence_inf_dt = batch_inf_dt / max(1, len(batch.records))

            for row_idx, record in enumerate(batch.records):
                trimmed = batch_log_probs[row_idx, 1:len(record.sequence) + 1, :]
                prepared = model._prepare_decode_inputs(trimmed)
                if active_worker_pool is None or prepared.sub_emits_cpu is None or prepared.none_sub is None:
                    decoded = model._decode_prepared_inputs(
                        prepared=prepared,
                        sequence=record.sequence,
                        beam_size=beam_size,
                        prior_mid_to_start=prior_mid_to_start,
                        prior_stop_to_mid=prior_stop_to_mid,
                        prior_stop_to_start=prior_stop_to_start,
                        refine_extended=refine_extended,
                        include_verbose_artifacts=False,
                    )
                    _store_completed(
                        order_idx=record.order_idx,
                        seq_id=record.seq_id,
                        sequence=record.sequence,
                        prepared=prepared,
                        decoded=decoded,
                        inf_dt=per_sequence_inf_dt,
                    )
                    continue

                task = _DecodeTask(
                    order_idx=record.order_idx,
                    seq_id=record.seq_id,
                    sequence=record.sequence,
                    prepared=prepared,
                    inf_dt=per_sequence_inf_dt,
                )
                pending_tasks[record.order_idx] = task
                bundle_buffer.append(task)
                _flush_bundle()
                if len(pending_tasks) > effective_max_queued_seqs:
                    _flush_bundle(force=True)
                    while len(pending_tasks) > effective_max_queued_seqs:
                        _drain_results(block=True)

            _flush_bundle(force=True)
            _drain_results(block=False)

        _flush_bundle(force=True)
        while pending_tasks:
            _drain_results(block=True)
    finally:
        pbar.close()
        if local_worker_pool is not None:
            local_worker_pool.close()

    ordered_results = [result for result in completed if result is not None]
    if len(ordered_results) != len(records):
        raise RuntimeError("Fast scan completed with missing sequence results.")

    rendered_output = ""
    if emit_console or emit_text_file:
        reports = [
            _render_report(
                model,
                result,
                score_thresh=score_thresh,
                evalue_thresh=evalue_thresh,
                refine_extended=refine_extended,
                beam_size=beam_size,
            )
            for result in ordered_results
        ]
        if reports:
            rendered_output = "\n\n".join(reports) + "\n"

    if emit_text_file and txt_path is not None:
        with open(txt_path, "w", encoding="utf-8") as handle:
            handle.write(rendered_output)
    if emit_console and rendered_output:
        print(rendered_output, end="")

    if to_tsv:
        rows: List[HitRow] = []
        for result in ordered_results:
            rows.extend(result.hit_rows)
        model._write_hits_tsv(to_tsv, rows)

    return {result.seq_id: result.domains_scored for result in ordered_results}
