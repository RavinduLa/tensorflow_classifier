import keras.models
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def check_config():
    print("Tensorflow version : ", tf.__version__)


def run_model():
    path = "C:\\Users\\HP\\PycharmProjects\\tensorflow_classifier\\models\\venic-model-v1\\venic-model-v1"
    print("Path : ", path)

    vocab_size = 10000
    embedding_dim = 16
    max_length = 100
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 1000

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    # loaded_model = tf.saved_model.load(path)
    loaded_model = keras.models.load_model(path)
    print(loaded_model)

    test_sentences = ["add a new property"]
    sentence = ["create a property named car",
                "checkout new branch named dev",
                "run the project",
                "iterate through the cars array",
                "what is the number of lines on the file",
                "are there any errors"]

    # very important but still has issues
    tokenizer.fit_on_texts(test_sentences)

    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    print(loaded_model.predict(padded))


if __name__ == '__main__':
    check_config()
    run_model()
