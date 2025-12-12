## 🚀 Project Setup Guide

Follow the steps below to set up the environment, prepare the dataset, and run the application.

### **1. Create a Virtual Environment**

```bash
python -m venv venv/
```

### **2. Activate the Virtual Environment**

```bash
venv/Scripts/activate
```

### **3. Install Required Packages**

```bash
pip install -r requirements.txt
```

### **4. Download & Prepare the Dataset**

Run the setup script to automatically download the **Food-41 dataset** into `data/raw/food41/`:

```bash
python src/setup_dataset.py
```

### **5. Split the Dataset**

This will split the raw dataset into **train**, **test**, and **validation** sets:

```bash
python src/split_dataset.py
```

### **6. Test the Data Loader**

This script will load sample batches and display an image visualization to confirm everything works correctly:

```bash
python src/data_loader.py
```

### **7. Launch the Streamlit App**

Start the user interface to test the model with uploaded images:

```bash
streamlit run app/streamlit_app.py
```

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
