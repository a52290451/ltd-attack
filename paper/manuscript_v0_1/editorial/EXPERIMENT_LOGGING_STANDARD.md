# LTD-Attack — Required live logging for future Perseo experiments

**Do not alter the currently running NetCLR process or frozen checkpoints.** Apply to subsequent scripts/runs only.

1. Every run prints a header immediately: run-id, UTC start, git SHA, script SHA, config SHA, CUDA availability, GPU name/VRAM, PID, counts and chronological partitions.
2. Structured JSON-lines file (`progress.jsonl`): one row per 20/50 training batches (configurable) with `epoch`, `batch`, `batches_total`, `samples_seen`, `loss_running_mean`, `elapsed_seconds`, `batches_per_second`, `gpu_mem_allocated_MiB`; flush every row.
3. One epoch summary with `epoch`, `loss`, `eval_metrics` (where authorized), elapsed time, epoch duration, learning rate; print to console (`flush=True`) and JSONL.
4. A lightweight heartbeat every 60 seconds (via batch-loop checks; no background daemons). CPU-heavy data preprocessing emits one progress item per 5,000 examples with time/throughput.
5. Checkpoint after complete epochs as declared; include RNG states and optimizer/scheduler/scaler if resume is supported. Use atomic write/rename and SHA256 manifest; clearly distinguish a resumed from a new run.
6. Explicit final status (`RUN_PASS` or `RUN_FAIL` with traceback), exit nonzero for failure, and preserve logs. Verify process and artifacts after an SSH disconnect.
7. No duplicate jobs: run lock or PID ownership validation, and refusal to overwrite any previous run. Cache creation and reuse must be content verified.
8. Monitoring commands: `tail -n 30 <log>`, `ps -p <pid> -o pid,stat,etime,%cpu,rss,args`, `nvidia-smi`, read-only manifest checks. Zero lines in log does **not** establish progress.
9. Do not introduce Future-B or Dataset C into fitting, unlabeled pretraining, checkpoint selection, or run-time configuration.
