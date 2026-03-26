# AI GPU Stress Test (RTX 5090 / Hopper / Datacenter GPUs)

## 🔧 Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

---

## ▶️ Run
```bash
python3 gpu-stress-test.py [OPTIONS]
```

### DESCRIPTION
Runs an AI-oriented GPU stress test using attention + matmul workloads.  
Outputs live metrics and generates structured reports on completion.

---

### OPTIONS
```bash
--duration <minutes>          Test duration (default: 20)

--gpu-class <type>            GPU class override
                              (auto | consumer | datacenter)

--profile <name>              Workload profile
                              (auto | consumer | datacenter | datacenter-max)

--target-vram-ratio <float>   Fraction of total VRAM to use (e.g. 0.8)

--target-vram-gb <int>        Absolute VRAM target (overrides ratio)

--dtype <type>                Precision
                              (bf16 | fp16)

--plain                       Line-oriented output (no dashboard)
```

---

### EXAMPLES
```bash
# Default run (20 minutes, auto-detect)
python3 gpu-stress-test.py

# Custom duration
python3 gpu-stress-test.py --duration 30

# RTX / consumer GPU
python3 gpu-stress-test.py --profile consumer

# H100 / datacenter GPU
python3 gpu-stress-test.py --profile datacenter

# Maximum saturation (H100+)
python3 gpu-stress-test.py --profile datacenter-max

# Custom memory usage
python3 gpu-stress-test.py --target-vram-ratio 0.82

# Plain output (for logs / CI)
python3 gpu-stress-test.py --plain
```
