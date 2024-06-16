import tensorflow as tf
import numpy as np
import string
import os

# Define necessary functions from preprocess.py here or ensure they are imported correctly
def load_data(filepath):
    with open(filepath, 'r') as file:
        text = file.read()
    return text

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    return text

# Load and preprocess data
data_path = 'data/input.txt'
text = load_data(data_path)
text = preprocess_text(text)

# Create character mappings including punctuation
vocab = sorted(set(text))
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = np.array(vocab)

# Load model and weights
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024
batch_size = 1

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(batch_shape=(batch_size, None)),
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size)
checkpoint_dir = 'checkpoints'

# Debugging: List files in the checkpoint directory
print("checkpoint directory contents:", os.listdir(checkpoint_dir))

latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
print("Latest checkpoint:", latest_checkpoint)

if latest_checkpoint:
    model.load_weights(latest_checkpoint)
    print(f"Loaded weights from {latest_checkpoint}")
else:
    print("No checkpoint found. Please ensure that the model has been trained and saved checkpoints.")

model.build(tf.TensorShape([batch_size, None]))

def generate_text(model, start_string, char2idx, idx2char, num_generate=1000, temperature=1.0):
    # Convert start_string to lowercase to match preprocessing
    start_string = start_string.lower()
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    # Reset states for each LSTM layer in the model
    for layer in model.layers:
        if hasattr(layer, 'reset_states'):
            layer.reset_states()

    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)

start_string = "To be, or not to be: "
generated_text = generate_text(model, start_string, char2idx, idx2char)
print(generated_text)
