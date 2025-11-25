FROM nvidia/cuda:12.6-devel-ubuntu22.04
RUN apt update && apt install -y python3-pip git
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "launch_demon.py", "--template", "templates/gemm_toon_v1.json", "--gpus", "0"]
