import magenta
import note_seq
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from magenta.models.transformer import transformer_model
from magenta.models.transformer import transformer_dataset
from magenta.models.transformer import transformer_utils

from utils.midi_utils import midi_to_note_sequence, note_sequence_to_midi

# Load the seed input
seed_path = 'seed.mid'
seed_sequence = midi_to_note_sequence(seed_path)

# Define model parameters
config = transformer_utils.get_transformer_config('melody_rnn')
config.hparams.batch_size = 1
checkpoint_path = 'model/model.ckpt'

# Initialize the model
model = transformer_model.TransformerModel(config, is_training=False, name='transformer')
inputs = transformer_dataset.get_dataset(config, is_training=False).make_one_shot_iterator().get_next()

# Run the generation
with tf.Session() as sess:
    model.initialize(sess)
    model.restore(sess, checkpoint_path)

    generated_seq = model.generate(
        sess,
        primer_sequence=seed_sequence,
        generate_length=128,
        temperature=1.0
    )

# Save the output
note_sequence_to_midi(generated_seq, 'output.mid')
print("Melody generated and saved to output.mid.")
