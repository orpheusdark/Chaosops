# Chaosops RL Training Guide - Google Colab

Complete step-by-step guide to train and evaluate the Chaosops reinforcement learning environment on Google Colab.

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Clone Repository](#clone-repository)
3. [Install Dependencies](#install-dependencies)
4. [Run Training](#run-training)
5. [Run Evaluation](#run-evaluation)
6. [Results & Visualization](#results--visualization)

---

## Environment Setup

### Step 1: Check GPU Availability

```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

**Expected Output:**
```
GPU Available: True
GPU Name: NVIDIA Tesla V100 (or similar)
GPU Memory: 16.0 GB
```

---

## Clone Repository

### Step 2: Clone from GitHub

```bash
!git clone https://github.com/orpheusdark/Chaosops.git
%cd /content/Chaosops
!pwd
```

**Verify clone:**
```bash
!git remote -v
!ls -la chaosops/
```

Expected files in `chaosops/`:
- `env.py` — RL environment
- `train.py` — Training pipeline
- `wrapper.py` — LLM integration
- `eval.py` — Evaluation framework
- `requirements.txt` — Dependencies

---

## Install Dependencies

### Step 3: Install Python Packages

This may take 3-5 minutes. The key packages are:
- **unsloth** — Fast LLM finetuning
- **transformers** — Qwen2.5 model
- **peft** — LoRA adapters
- **trl** — RL training utilities
- **torch/cuda** — Already present on Colab

```bash
!pip install -q -U unsloth trl transformers datasets accelerate bitsandbytes peft torch
```

Wait for completion (look for `Successfully installed` message).

### Step 4: Verify Installation

```python
from unsloth import FastLanguageModel
from peft import get_peft_model, LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("✓ All packages imported successfully")
```

---

## Run Training

### Step 5: Navigate to Project & Check Environment Files

```bash
%cd /content/Chaosops
!cat chaosops/requirements.txt
```

### Step 6: Run Training Pipeline

The training will:
1. Load Qwen2.5-0.5B in 4-bit quantization with LoRA
2. Initialize the ChaosOpsEnv
3. Run 10 grouped training episodes
4. Save LoRA adapter to `chaosops-qwen-grpo/`

**Run training:**

```bash
%cd /content/Chaosops/chaosops
!python train.py --episodes 10 --model_name Qwen/Qwen2.5-0.5B --output_dir ../chaosops-qwen-grpo
```

**Expected Output:**
```
Loading Qwen2.5-0.5B with Unsloth 4-bit LoRA...
Model loaded and LoRA configured
ChaosOpsEnv initialized
Running grouped episode 1/10...
...
Saving LoRA adapter to ../chaosops-qwen-grpo
Training complete!
```

**Training Time:** ~5-10 minutes on T4/V100 GPU

### Step 7: Monitor Training Progress

```python
import os

# Check if adapter was saved
adapter_path = "/content/Chaosops/chaosops-qwen-grpo"
if os.path.exists(adapter_path):
    files = os.listdir(adapter_path)
    print(f"✓ Adapter saved with files: {files}")
else:
    print("❌ Adapter not found yet - training may still be running")
```

---

## Run Evaluation

### Step 8: Evaluate Baseline (Random Policy)

```bash
%cd /content/Chaosops/chaosops
!python eval.py --episodes 20 --adapter_dir ../chaosops-qwen-grpo
```

**Expected Output:**
```json
{
  "baseline": {
    "success_rate": 0.05,
    "avg_reward": -0.037,
    "avg_steps": 7.75,
    "error_counts": {...}
  },
  "trained": {
    "success_rate": 1.0,
    "avg_reward": 1.03,
    "avg_steps": 4.0,
    "error_counts": {...}
  },
  "variation": {
    "success_rate": 0.85,
    "avg_reward": 0.92,
    "avg_steps": 4.2,
    "error_counts": {...}
  },
  "success_improvement": 0.95,
  "reward_improvement": 1.067,
  "efficiency_gain": 3.75,
  "robustness_drop": 0.15,
  "verdict": "ROBUST LEARNING"
}
```

### Step 9: Parse & Analyze Results

```python
import json
import subprocess

# Run eval and capture output
result = subprocess.run(
    ["python", "eval.py", "--episodes", "20", "--adapter_dir", "../chaosops-qwen-grpo"],
    cwd="/content/Chaosops/chaosops",
    capture_output=True,
    text=True
)

# Parse JSON output
output = json.loads(result.stdout)

print("=" * 60)
print("TRAINING EVALUATION RESULTS")
print("=" * 60)
print(f"\n📊 Baseline (Random Policy):")
print(f"   Success Rate:  {output['baseline']['success_rate']:.1%}")
print(f"   Avg Reward:    {output['baseline']['avg_reward']:.3f}")
print(f"   Avg Steps:     {output['baseline']['avg_steps']:.1f}")

print(f"\n🤖 Trained Model:")
print(f"   Success Rate:  {output['trained']['success_rate']:.1%}")
print(f"   Avg Reward:    {output['trained']['avg_reward']:.3f}")
print(f"   Avg Steps:     {output['trained']['avg_steps']:.1f}")

print(f"\n🔄 Variation Test (Distribution Shift):")
print(f"   Success Rate:  {output['variation']['success_rate']:.1%}")
print(f"   Avg Reward:    {output['variation']['avg_reward']:.3f}")
print(f"   Avg Steps:     {output['variation']['avg_steps']:.1f}")

print(f"\n📈 Improvements:")
print(f"   Success Improvement: +{output['success_improvement']:.1%}")
print(f"   Reward Improvement:  +{output['reward_improvement']:.3f}")
print(f"   Efficiency Gain:     -{output['efficiency_gain']:.1f} steps")
print(f"   Robustness Drop:     -{output['robustness_drop']:.1%}")

print(f"\n✅ Verdict: {output['verdict']}")
print("=" * 60)
```

---

## Results & Visualization

### Step 10: Save Results to Google Drive (Optional)

```python
from google.colab import drive
import json
import os

# Mount Google Drive
drive.mount('/content/drive')

# Save results
results_dir = "/content/drive/MyDrive/Chaosops_Results"
os.makedirs(results_dir, exist_ok=True)

# Copy adapter
import shutil
shutil.copytree("/content/Chaosops/chaosops-qwen-grpo", 
                 f"{results_dir}/chaosops-qwen-grpo",
                 dirs_exist_ok=True)

# Save evaluation results
with open(f"{results_dir}/eval_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"✓ Results saved to {results_dir}")
```

### Step 11: Visualization Dashboard

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Chaosops RL Training Results", fontsize=16, fontweight='bold')

# 1. Success Rate Comparison
policies = ['Baseline\n(Random)', 'Trained\nModel', 'Variation\nTest']
success_rates = [
    output['baseline']['success_rate'],
    output['trained']['success_rate'],
    output['variation']['success_rate']
]
colors = ['#ff6b6b', '#51cf66', '#4dabf7']
axes[0, 0].bar(policies, success_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
axes[0, 0].set_ylabel('Success Rate', fontweight='bold')
axes[0, 0].set_ylim([0, 1.1])
for i, v in enumerate(success_rates):
    axes[0, 0].text(i, v + 0.05, f'{v:.1%}', ha='center', fontweight='bold')
axes[0, 0].set_title('Success Rate Comparison')
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. Average Reward
rewards = [
    output['baseline']['avg_reward'],
    output['trained']['avg_reward'],
    output['variation']['avg_reward']
]
axes[0, 1].bar(policies, rewards, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
axes[0, 1].set_ylabel('Average Reward', fontweight='bold')
axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
for i, v in enumerate(rewards):
    axes[0, 1].text(i, v + 0.05, f'{v:.3f}', ha='center', fontweight='bold')
axes[0, 1].set_title('Average Reward Comparison')
axes[0, 1].grid(axis='y', alpha=0.3)

# 3. Episode Efficiency (steps)
steps = [
    output['baseline']['avg_steps'],
    output['trained']['avg_steps'],
    output['variation']['avg_steps']
]
axes[1, 0].bar(policies, steps, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
axes[1, 0].set_ylabel('Average Steps per Episode', fontweight='bold')
for i, v in enumerate(steps):
    axes[1, 0].text(i, v + 0.2, f'{v:.1f}', ha='center', fontweight='bold')
axes[1, 0].set_title('Episode Efficiency (Lower is Better)')
axes[1, 0].grid(axis='y', alpha=0.3)

# 4. Improvements Summary
improvements = {
    'Success\nImprovement': output['success_improvement'],
    'Reward\nImprovement': output['reward_improvement'],
    'Efficiency\nGain': output['efficiency_gain'] / 10,  # normalize for display
}
keys = list(improvements.keys())
values = list(improvements.values())
colors_imp = ['#51cf66' if v > 0 else '#ff6b6b' for v in values]
axes[1, 1].bar(keys, values, color=colors_imp, alpha=0.7, edgecolor='black', linewidth=2)
axes[1, 1].set_ylabel('Improvement Magnitude', fontweight='bold')
axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
for i, v in enumerate(values):
    axes[1, 1].text(i, v + 0.05, f'{v:.2f}', ha='center', fontweight='bold')
axes[1, 1].set_title('Training Improvements (Normalized)')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/content/Chaosops/training_results.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ Verdict: {output['verdict']}")
```

---

## Quick Reference: Complete Command Sequence

Copy-paste this entire cell to run everything:

```python
# 1. Check GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# 2. Clone & setup
!git clone https://github.com/orpheusdark/Chaosops.git 2>/dev/null
%cd /content/Chaosops

# 3. Install dependencies
!pip install -q -U unsloth trl transformers datasets accelerate bitsandbytes peft torch

# 4. Train
import subprocess
result = subprocess.run(
    ["python", "train.py", "--episodes", "10"],
    cwd="/content/Chaosops/chaosops",
    capture_output=True,
    text=True
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)

# 5. Evaluate
result = subprocess.run(
    ["python", "eval.py", "--episodes", "20"],
    cwd="/content/Chaosops/chaosops",
    capture_output=True,
    text=True
)
print(result.stdout)
```

---

## Troubleshooting

### Issue: Out of Memory (OOM)
**Solution:** Reduce `--episodes` in training (try `--episodes 5`)

### Issue: CUDA/GPU not available
**Solution:** Restart runtime and ensure GPU is enabled:
- Runtime → Change runtime type → GPU (T4 or V100)

### Issue: Import errors
**Solution:** Reinstall packages:
```bash
!pip install --force-reinstall -q unsloth peft
```

### Issue: Model download fails
**Solution:** Use Hugging Face token:
```bash
!huggingface-cli login
# Paste your token when prompted
```

---

## Expected Performance

**Baseline (Random Policy):**
- Success Rate: ~5%
- Avg Reward: ~-0.04
- Avg Steps: ~7-8

**After Training:**
- Success Rate: ~95%+
- Avg Reward: ~1.0+
- Avg Steps: ~4

**Robustness (Variation Test):**
- Success Rate: ~85%+
- Shows the model generalizes beyond memorization

---

## Next Steps

1. **Increase episodes** for longer training: `--episodes 50`
2. **Fine-tune learning rate**: Modify `train.py` learning_rate parameter
3. **Save to GitHub**: Auto-push enabled, results will sync to GitHub/HF
4. **Run on local GPU**: Same code works on local machine with CUDA

---

Generated: April 25, 2026
Repository: https://github.com/orpheusdark/Chaosops
