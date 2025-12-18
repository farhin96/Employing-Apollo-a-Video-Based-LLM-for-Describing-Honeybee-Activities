```shell
pip install -r requirements.txt
Bee Activity QA 

This project classifies honey-bee behavior from MP4 videos (fanning / grooming / signaling) using an internal OpenCV module for robust labeling and a fine-tuned Apollo-3B multimodal model for grounded natural-language explanations. A Gradio UI provides interactive video upload + Q/A.

---

## Requirements

### Hardware (recommended)
- **GPU (CUDA)**: Optional but strongly recommended for speed  
- **CPU-only**: Works, but inference will be much slower

### Software
- **Python**: 3.9–3.11
- **OS**: Windows / Linux / macOS (Windows supported)

### Python packages
Core dependencies:
- `torch`
- `transformers`
- `huggingface_hub`
- `gradio`
- `opencv-python`
- `numpy`

Apollo project utilities (must exist in your repo):
- `utils/mm_utils.py`
- `utils/conversation.py`
- `utils/constants.py`

Model/checkpoint:
- Base model: `GoodiesHere/Apollo-LMMs-Apollo-3B-t32` 
- Fine-tuned checkpoint: `checkpoints/apollo_bee_vqa_best_cpu.pth` 

---

## Setup

### 1) Create environment (recommended)
**Conda**
```bash
conda create -n bees python=3.10 -y
conda activate bees

2) Install dependencies

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers huggingface_hub gradio opencv-python numpy

3) Running the App
python bee_app.py

By default it launches locally at:

http://127.0.0.1:7860



