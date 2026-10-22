# CUDAgraveyard ⚰️

**The final resting place for obsolete CUDA kernels.**  
Autonomous AI that discovers faster, more energy-efficient kernels than any human engineer – burying cuBLAS, FlashAttention, and Triton in eternal obsolescence. No mercy, no weekends, just relentless optimization loops.

> **Status: concept and roadmap.** Nothing has been measured yet. No kernel in
> this repository is benchmarked against cuBLAS, FlashAttention or Triton, and
> no comparison run exists. The body count below is a list of targets, not of
> results.

## Body Count (November 2025 – 2026 Goals)
- [ ] GEMM 8k×8k (vs cuBLAS Lt 12.6) – Target: +25% TFLOPS, -40% watts on H100
- [ ] FlashAttention-3 kernels (vs xformers) – Roofline domination
- [ ] Llama.cpp GGUF inference (Q4_K_M + imatrix) – CPU+GPU hybrid speedup
- [ ] Custom LLVM passes for SPECfp – Auto-discover unrolling magic

Each kill includes:
- Full `.cu` source + git patch
- Benchmark tables (TFLOPS, latency, power draw)
- Flamegraphs + nsight reports
- Arrogant auto-blogpost: "We just obsoleted 5 years of NVIDIA sweat."

## Quickstart

### Current validation boundary

This repository is an experimental prototype. Run
`python scripts/validate_offline.py` to validate Python syntax, reject embedded
shell wrappers and check JSON template structure without installing packages,
calling an API or starting a GPU workload. CI runs only these offline checks.

The CUDA benchmark path is not qualified: kernel assembly, iteration parsing,
automatic matrix sizing and correctness checks still need runtime work. A
passing offline check establishes no speedup, energy saving or cuBLAS result.
The former `benchmark.yml` contained prose rather than a workflow and has been
removed. GPU experiments remain a separate manual operation.

1. Clone: `git clone https://github.com/zedarvates/CUDAgraveyard.git`
2. Setup: `pip install -r requirements.txt` (a code-generation model API key, shinka-evolve, nsight-sdk)
3. Launch a demon: `python launch_demon.py --template templates/gemm_toon_v1.json --gpus 0`
4. Watch the graveyard fill: Logs in `graves/` (dead kernels + their epitaphs).

## Stack (2025 Edition)
- **Core**: AI-Scientist-v2 (SakanaAI fork) + ShinkaEvolve for genetic mutations
- **Tools**: nvcc 12.6+, nsight-compute, hyperfine, nvidia-smi power monitoring
- **AI Brain**: any reasoning code-generation model, supplied by configuration (no vendor is hard-coded)
- **Hardware**: intended for H100/L40S/MI300X; Docker for sandboxed runs
- **Safety**: Strict timeouts + ethical watcher (no infinite loops or self-hacks)

**Warning**: an unattended search pins a GPU for hours — set timeouts and watch power draw and temperature. Overclocking is your own business.

## Roadmap to Armageddon
- Q4 2025: GEMM graveyard opening
- Q1 2026: Multi-kernel massacre (attention + transformers)
- Q2 2026: Hybrid CPU-GPU-Quantique integration (link to CosmosCluster)
- Beyond: Self-optimizing the optimizer (Darwin Gödel vibes)

Fork it, contribute kills, or just watch the jobs vanish.  
*Built by zedarvates – because why hire when AI slays?*


---

[![Donate](https://img.shields.io/badge/☕%20Soutenir-BTC%20%7C%20ETH-orange)](DONATE.md)
