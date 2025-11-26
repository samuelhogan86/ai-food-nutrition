run python src/setup_dataset.py

---

this will download food41 into data/raw/food41/

---

## GPU / Accelerator setup

It is best practice to run AI models on GPU. CUDA (for Windows/Linux) and MPS (for MacOS) help take advantage of GPU. Below is how you can check/download your respective platform.

**Notes:**
- CUDA: installing a CUDA-enabled PyTorch wheel provides the CUDA runtime libraries, but you still need a compatible NVIDIA driver installed on the OS. Use `nvidia-smi` to confirm driver + GPU availability.

- MPS: macOS uses Apple Metal (MPS); recent official PyTorch macOS wheels include MPS support and do not need separate GPU drivers.

### Quick checks
- Check accelerator support from Python:

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"
python -c "import torch; print('mps:', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())"
```

- OS-level checks:
  - Linux/Windows (NVIDIA): `nvidia-smi`
  - macOS: `system_profiler SPDisplaysDataType` or Activity Monitor → GPU

### Install PyTorch (pick the right command at https://pytorch.org/get-started/locally/)

Common examples:

- Conda + CUDA (example, CUDA 11.8):

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

- Pip + CUDA (example, CUDA 11.8):

```bash
pip3 install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

- macOS (MPS / Apple GPU):

```bash
conda install pytorch torchvision -c pytorch
# or follow the macOS install instructions on the PyTorch site
```

### Links
- PyTorch local install helper: https://pytorch.org/get-started/locally/
- NVIDIA driver downloads: https://www.nvidia.com/Download/index.aspx