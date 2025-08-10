from transformers import AutoModelForCausalLM
from midi2audio import FluidSynth
from anticipation.sample import generate
from anticipation.convert import events_to_midi

# Load model
model = AutoModelForCausalLM.from_pretrained("stanford-crfm/music-small-800k")

# SoundFont path on macOS
soundfont_path = "/Library/Audio/Sounds/Banks/FluidR3_GM.sf2"
fs = FluidSynth(soundfont_path)

# Generate 10 seconds of music
tokens = generate(model, start_time=0, end_time=10, top_p=0.98)

# Save to MIDI
midi_path = "output.mid"
mid = events_to_midi(tokens)
mid.save(midi_path)

# Synthesize audio
fs.midi_to_audio(midi_path, "output.wav")

print("Done! Your audio is saved as 'output.wav'")

