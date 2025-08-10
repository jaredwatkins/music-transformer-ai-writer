# Human-AI Co-Creation in Melody Writing

This project explores interactive co-creation between human musicians and an AI model. Using the **Stanford CRFM Anticipatory Music Transformer** (`music-small-800k`), the system generates symbolic music continuations from a seed melody and allows iterative prompting.

### Model
- [`stanford-crfm/music-small-800k`](https://huggingface.co/stanford-crfm/music-small-800k)
- Autoregressive transformer trained on 800k MIDI sequences
- Hosted via HuggingFace Transformers

###️ Features
- Interactive loop-based co-writing: Accept or reject each proposed continuation
- Model-to-MIDI conversion using `anticipation` repo utilities
- Optional audio synthesis via FluidSynth
- Seed + continuation saved as `output.mid`

---

##  Setup

### 1. Clone and install

```bash
git clone https://github.com/jaredwatkins/music-transformer-ai-writer.git
cd music-transformer-ai-writer
python3.8 -m venv venv-py38
source venv-py38/bin/activate
pip install -r anticipation/requirements.txt# Music Transformer AI Writer

This project demonstrates how to use the Music Transformer model to generate melody continuations from a short seed input. It uses Magenta and TensorFlow 1.x.

## How to Use

1. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```

2. Place a 4-bar seed MIDI file as `seed.mid`

3. Run:
    ```bash
    python generate.py
    ```

4. The generated melody will be saved as `output.mid`.

## Notes

- You will need the Music Transformer checkpoint in `model/model.ckpt`
- Tested with Python 3.8
