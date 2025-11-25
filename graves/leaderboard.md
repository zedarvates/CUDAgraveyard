ID,Famille de kernels,Usages dominants,Hardware principal,Objectif démon 2026-2027,Difficulté (1-10),Commentaires / idées de mutations spécifiques
G01,"GEMM classique (SGEMM, DGEMM, HGEMM, BF16, TF32, FP8)","LLMs, diffusion, training, HPC",NVIDIA H100/L40S/MI300X,+30 % TFLOPS + -45 % watts vs cuBLAS/Triton,7,cluster launch + swizzle 3D + async mma pipelines
G02,Batched GEMM & GEMM strided,"Transformer layers, grouped conv",NVIDIA + AMD,battre TensorRT-LLM et rocBLAS simultanément,8,auto-tune stride via genetic search
A01,FlashAttention-2/3 + FlashDecoding,"Inference LLM (Llama, Mistral, DeepSeek)",NVIDIA 40xx/H100,+50 % tokens/s + -60 % énergie vs xFormers,9,paged KV cache + speculative decoding kernels
A02,PagedAttention + GQA/MQA kernels,"vLLM, TGI, Ollama",NVIDIA + AMD ROCm,rendre vLLM obsolète,9,block-sparse attention auto-discovered
A03,Speculative decoding kernels,"Medusa, Lookahead, Eagle",NVIDIA,+100 % tokens/s effectifs,10,multi-draft tree kernels
T01,Triton hand-written kernels (tous les tops GitHub),Toute la scène Triton 2024-2025,NVIDIA,battre chaque kernel du top 50 Triton leaderboard,8,le démon lit le repo Triton et les enterre un par un
D01,Diffusion U-Net kernels (conv + attention),"Stable Diffusion XL, Flux, SD3",NVIDIA + AMD + Intel Arc,2× it/s vs Automatic1111 + ComfyUI,8,"fused conv+gelu+layernorm, FP8 mixed"
D02,VAE encode/decode,"SDXL, SD3, Flux",NVIDIA,battre les kernels xFormers VAE de 300 %,7,
P01,N-body / Barnes-Hut / Fast Multipole,"Simulations astro, moléculaires",NVIDIA + futur PhysX PCIe,remplacer GPU + cartes physiques dédiées,9,lien direct avec CosmosCluster
P02,SPH (Smoothed Particle Hydrodynamics),"Fluides, explosions demoscene",NVIDIA,temps réel 10 M particules sur une seule 5090,8,
P03,Rigid body / constraint solvers,"Physiques jeux, robotics",NVIDIA + cartes PhysX-like,remplacer PhysX/NVIDIA PhysX complet,10,futur cible quand les cartes PCIe reviennent
C01,"Cryptographie (Keccak, Blake3, SHA3)","Mining, zero-knowledge proofs",NVIDIA + AMD,battre les mineurs ASIC avec du CUDA pur,9,kernels 256-bit vectorisés
C02,Ray-tracing BVH build + traversal,"Path tracing, RTX",NVIDIA RTX,battre OptiX de 30 % sans hardware RT,10,
V01,"Video encoding (AV1, HEVC, VVC) kernels",FFmpeg x264/x265 → GPU,NVIDIA NVENC,remplacer NVENC par du compute pur 20 % plus rapide,9,
Q01,Hybride classique → quantique kernels,Pré-traitement pour algos quantiques,NVIDIA + IBM Quantum link,auto-convertir parties Monte-Carlo en Grover/QAOA,10+,phase CosmosCluster 2028+
S01,Size-coding / 4k/8k/64k demoscene kernels,"Demoscene, shadertoy extrême",Toute carte 2015-2025,générer des 4k qui gagnent Revision automatiquement,10,le démon devient le nouveau IQ/RGBA