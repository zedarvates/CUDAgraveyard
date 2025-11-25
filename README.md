# CUDAgraveyard ⚰️

Autonomous optimization demon that kills hand-tuned CUDA kernels and puts them in the graveyard.

Current body count (2025–2026) :
- [ ] cuBLAS GEMM (H100)
- [ ] FlashAttention-3
- [ ] TensorRT-LLM layer kernels
- [ ] xFormers / Triton implementations

Every kernel that dies here gets :
- Full source + patch
- Benchmark table vs official baseline
- Power trace (watts saved)
- Arrogant blogpost auto-published

Fully based on AI-Scientist-v2 + ShinkaEvolve + nsight-compute + real H100/L40S/MI300X hardware.

License: Apache 2.0 (with the clause "you assume if your datacenter melts")
