import tensorflow as tf
import numpy as np
from preprocess import load_data, preprocess_text, create_sequences
from model import build_model
import os

# Load and preprocess data
data_path = 'data/input.txt'
text = load_data(data_path)
text = preprocess_text(text)

# Create character mappings
vocab = sorted(set(text))
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = np.array(vocab)

# Create sequences
seq_length = 100
dataset = create_sequences(text, seq_length, char2idx)

# Prepare data for training
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Build the model
embedding_dim = 256
rnn_units = 1024
vocab_size = len(vocab)
model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)

# Compile the model
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

# Train the model
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt-{epoch}.weights.h5")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True,
    monitor = 'val_accuracy',
    mode = 'max',
    verbose=1  # Added verbosity to check if the callback is being triggered
)
EPOCHS = 3
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
