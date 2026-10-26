# Week 1 Checklist — Carter Offline Supply Chain

**Date started:** _______________  
**Target completion:** Before final air-gap  
**Sign-off:** _______________

> Complete every item in this checklist before disconnecting the network.
> The machine must be able to install and run Carter from local media only.

---

## 1. Storage and Directory Setup

- [ ] Confirm available SSD space (need 30–60 GB free)
  - Available: _______________ GB
- [ ] Create the Carter project directory at the target location
- [ ] Create a dedicated `wheelhouse/` directory on external storage
- [ ] Confirm an external backup drive is available for the offline kit

---

## 2. Model Selection

### Primary Model (1B–1.5B instruct, GGUF 4-bit)
Recommended options (verify license before downloading):

| Model | Size (Q4_K_M) | License | VRAM req |
|-------|--------------|---------|----------|
| Llama 3.2 1B Instruct | ~0.7 GB | Meta Llama 3.2 | ~1 GB |
| Qwen2.5 1.5B Instruct | ~1.0 GB | Apache 2.0 | ~1.5 GB |
| SmolLM2 1.7B Instruct | ~1.0 GB | Apache 2.0 | ~1.5 GB |
| Phi-3.5 Mini Instruct | ~2.2 GB | MIT | ~3 GB |

- [ ] Selected primary model: ___________________________
- [ ] License reviewed and noted: ___________________________
- [ ] GGUF file downloaded: ___________________________
- [ ] Tokenizer files downloaded (if separate): ___________________________
- [ ] SHA256 checksum verified: ___________________________

### Fallback Model (smaller, for OOM situations)
- [ ] Selected fallback model: ___________________________
- [ ] Fallback model downloaded and checksummed

---

## 3. Python and Runtime

- [ ] Python 3.12.x Windows installer downloaded
  - URL: https://www.python.org/ftp/python/3.12.x/python-3.12.x-amd64.exe
  - Version: _______________
  - SHA256: _______________
- [ ] pip wheel for setuptools downloaded
- [ ] pip wheel for virtualenv or venv-related tools downloaded

---

## 4. Dependency Wheelhouse

Run these commands on a connected machine to build the wheelhouse:

```powershell
# Create a staging directory
mkdir carter-wheelhouse

# Download all wheels for requirements.txt
pip download -r requirements.txt -d ./carter-wheelhouse/ --platform win_amd64 --python-version 312 --only-binary=:all:

# Download llama-cpp-python (CUDA 12.x build for RTX 3050)
# Check https://github.com/abetlen/llama-cpp-python/releases for the correct wheel
# pip download llama-cpp-python==<version> --extra-index-url <cuda-whl-url> -d ./carter-wheelhouse/

# PyTorch (CPU) - download from official
# pip download torch torchvision --index-url https://download.pytorch.org/whl/cpu -d ./carter-wheelhouse/
# OR PyTorch (CUDA 12.1) for GPU training:
# pip download torch torchvision --index-url https://download.pytorch.org/whl/cu121 -d ./carter-wheelhouse/
```

- [ ] `requirements.txt` wheels downloaded to wheelhouse
- [ ] `llama-cpp-python` wheel downloaded (CUDA build verified)
- [ ] `torch` wheel downloaded (CPU or CUDA, as appropriate)
- [ ] Test installation in a clean virtual environment:
  ```powershell
  python -m venv test_venv
  test_venv\Scripts\activate
  pip install --no-index --find-links=./carter-wheelhouse/ -r requirements.txt
  python -c "import sklearn, numpy, pandas, yaml, click; print('OK')"
  ```
- [ ] Clean installation test PASSED

---

## 5. llama.cpp Server (optional — for HTTP inference endpoint)

- [ ] llama.cpp Windows release binary downloaded
  - URL: https://github.com/ggerganov/llama.cpp/releases
  - Version: _______________
  - Build: CUDA / CPU: _______________
- [ ] Binary tested locally: `llama-server --version`
- [ ] SHA256 checksum noted: _______________

---

## 6. Dataset for ML Model (Week 4)

Select a small, publicly available classification dataset:

Recommended options:
- UCI Breast Cancer Wisconsin (sklearn built-in) — no download needed
- UCI Iris (sklearn built-in) — no download needed
- Kaggle Credit Card Fraud (requires download before air-gap)

- [ ] Dataset selected: ___________________________
- [ ] Dataset downloaded and placed in `data/raw/` (if not sklearn built-in)
- [ ] Dataset license reviewed: ___________________________
- [ ] Data hash/provenance documented: ___________________________

---

## 7. Checksums and Manifest

- [ ] Generate SHA256 checksums for all model files:
  ```powershell
  Get-FileHash models\llm\*.gguf -Algorithm SHA256
  ```
- [ ] Record checksums in `docs/MANIFEST.md`
- [ ] Record checksums for wheelhouse zip archive
- [ ] Record Python installer checksum

---

## 8. Offline Kit Package

- [ ] Copy the entire Carter project directory to external storage
- [ ] Copy the wheelhouse to external storage
- [ ] Copy model files to external storage
- [ ] Copy Python installer to external storage
- [ ] Verify the external storage copy is readable
- [ ] Create a second backup on a separate drive or location

---

## 9. Architecture Baseline

- [ ] Confirm `configs/carter.example.yaml` has correct settings for this hardware
- [ ] Run `python -m carter.runtime.readiness` — document current status
- [ ] Initialize git repository: `git init && git add . && git commit -m "Week 1: Project scaffold"`
- [ ] Document selected model, VRAM budget, and context length in `configs/carter.yaml`

---

## 10. Sign-off

All items above are complete. The offline kit is verified on external storage.
The machine is ready to be air-gapped.

**Signed:** _______________ **Date:** _______________

---

## References

- llama.cpp releases: https://github.com/ggerganov/llama.cpp/releases
- llama-cpp-python CUDA wheels: https://github.com/abetlen/llama-cpp-python/releases
- PyTorch download: https://pytorch.org/get-started/locally/
- Hugging Face GGUF models: https://huggingface.co/models?library=gguf
- Python 3.12: https://www.python.org/downloads/release/python-3120/
