# Music Transformer AI Writer

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
