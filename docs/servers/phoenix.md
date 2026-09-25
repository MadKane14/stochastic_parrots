# Machine rules — RRC server `phoenix.rrcx.tk` (MUST follow in every session on this machine)

Install on phoenix as the user-level Claude Code memory so it applies to every session:
`mkdir -p ~/.claude && cp docs/servers/phoenix.md ~/.claude/CLAUDE.md`

This is a **shared** lab server with a strike policy (5 strikes ⇒ removed from the server). Breaking these
rules costs the user real penalties, so they override convenience and speed. If a task would break any rule,
stop and ask the user instead.

## Machine
- AMD Ryzen 9 7950X (16 cores / 32 threads), 128 GB RAM, 3× NVIDIA RTX A4000 (16 GB each), 1 TB NVMe (OS) +
  4 TB HDD scratch. No GUI. Other users run jobs here at the same time.
- User: `madhur.kankane`. Scratch space: `/scratch2/madhur.kankane/` (conda envs already live there).

## Storage (hard rules)
- **Never store or extract datasets, caches, checkpoints, embeddings or large outputs in `$HOME`.**
  `$HOME` is a 50 GB NFS quota synced across servers and slow. Code and git repos may stay in `$HOME`.
- Put all data and artifacts under `/scratch2/madhur.kankane/`:
  - data: `/scratch2/madhur.kankane/ber_data/` (→ `BER_DATA_DIR=/scratch2/madhur.kankane/ber_data/dataset`)
  - artifacts/caches/submissions: `BER_ARTIFACTS=/scratch2/madhur.kankane/ber_artifacts`
  - model caches: `HF_HOME=/scratch2/madhur.kankane/hf_cache`, `TORCH_HOME=/scratch2/madhur.kankane/torch_cache`
  - temp files: `TMPDIR=/scratch2/madhur.kankane/tmp`
- **Never fill `/scratch`/`/scratch2`.** Check `df -h /scratch2` before writing anything > 10 GB and keep our total
  footprint modest (target < 100 GB). Delete stale caches/runs you created when they are no longer needed.
- Never write outside `$HOME`, `/scratch2/madhur.kankane/` and `/tmp` (small temp files only).

## CPU / RAM / GPU (hard rules — "don't take 100% of the system")
- **Always set resource-control flags.** Every job: `BER_THREADS` ≤ 12 (the repo's run.py applies it to
  OMP/MKL/OpenBLAS/rapidfuzz/LightGBM), and run long jobs with `nice -n 10`. Never use "all cores" settings
  (`n_jobs=-1`, `workers=-1`, `num_workers=os.cpu_count()`), never launch several heavy jobs in parallel.
- RAM: keep a single job ≤ 64 GB (half the machine). Watch it with `free -g` / `htop`; if a job's memory keeps
  growing past that, kill it and make the code leaner. For CPU jobs you may cap memory with
  `systemd-run --user --scope -p MemoryMax=64G <cmd>` if available, or `ulimit -v` (not for GPU jobs).
- GPU: before any GPU job run `nvidia-smi`; use **one** free GPU via `CUDA_VISIBLE_DEVICES=<id>`; never grab all three.
  Keep batch sizes within 16 GB. Stop GPU processes when done.
- Check load before starting anything heavy: `uptime`, `htop`, `nvidia-smi`. If the machine is already busy
  (load average > ~20 or GPUs full), use fewer threads or wait, and tell the user.
- Long jobs: run inside `tmux` (session name `ber`) with logs to `$BER_ARTIFACTS/logs/`, and tell the user how
  to follow them. No orphaned background processes: list what you started and stop it when done.

## Software
- Prefer **conda** (conda-forge) for packages; use pip only if conda cannot provide the package, and only inside
  the `ber` env (`/scratch2/madhur.kankane/.conda/envs/ber`). Never `sudo`, never install system packages,
  never modify shared conda environments or anything under `/scratch/shared` or `/scratch/ros_workspaces`.
- No Docker on this server.

## Access etiquette (for the human user; don't automate)
- No NoMachine/GUI use needed for this project. Disconnect SSH with `logout`.
