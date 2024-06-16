import tensorflow as tf 
import numpy as np 
import string 

def load_data(filepath):
    with open(filepath, 'r') as file:
        text = file.read()
    return text 

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def create_sequences(text, seq_length, char2idx):
    text_as_int = np.array([char2idx[char] for char in text])
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text 

    dataset = sequences.map(split_input_target) 
    return dataset 