# coding: utf-8

# In[1]:

import re
import time
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors

print('TensorFlow Version: {}'.format(tf.__version__))

# !pip install tensorflow==1.1


REVIEWS = pd.read_csv("Reviews.csv")

REVIEWS = REVIEWS.dropna()
REVIEWS = REVIEWS.drop(['Id', 'ProductId', 'UserId', 'ProfileName',
                        'HelpfulnessNumerator', 'HelpfulnessDenominator',
                        'Score', 'Time'], 1)
REVIEWS = REVIEWS.reset_index(drop=True)

contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are"
}


def clean_text(text, remove_stopwords=True):
    """Remove unwanted characters, stopwords, and format the t
        ext to create fewer nulls word embeddings
    """

    # Convert words to lower case
    text = text.lower()

    # Replace contractions with their longer forms 
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)

    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)

    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    return text


# Clean the summaries and texts
CLEAN_SUMMARIES = []
for summary in REVIEWS.Summary:
    CLEAN_SUMMARIES.append(CLEAN_SUMMARIES(summary, remove_stopwords=False))
print("Summaries are complete.")

CLEAN_TEXTS = []
for text in REVIEWS.Text:
    CLEAN_TEXTS.append(CLEAN_TEXTS(text))
print("Texts are complete.")


def count_words(count_dict, text):
    """Count the number of occurrences of each word in a set of text"""
    for sentence in text:
        for word in sentence.split():
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1


WORD_COUNTS = {}

count_words(WORD_COUNTS, CLEAN_SUMMARIES)
count_words(WORD_COUNTS, CLEAN_TEXTS)

print("Size of Vocabulary:", len(WORD_COUNTS))

EMBEDDING_INDEX = {}
with open('numberbatch-en-19.08.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        EMBEDDING_INDEX[word] = embedding

print('Word embeddings:', len(EMBEDDING_INDEX))

MISSING_WORDS = 0
threshold = 20

for word, count in WORD_COUNTS.items():
    if count > threshold:
        if word not in EMBEDDING_INDEX:
            MISSING_WORDS += 1

MISSING_RATIO = round(MISSING_WORDS / len(WORD_COUNTS), 4) * 100

print("Number of words missing from CN:", MISSING_WORDS)
print("Percent of words that are missing from vocabulary: {}%".format(MISSING_RATIO))

# Limit the vocab that we will use to words that appear ≥ threshold or are in GloVe

# dictionary to convert words to integers
VOCAB_TO_INT = {}

value = 0
for word, count in WORD_COUNTS.items():
    if count >= threshold or word in EMBEDDING_INDEX:
        VOCAB_TO_INT[word] = value
        value += 1

# Special tokens that will be added to our vocab
codes = ["<UNK>", "<PAD>", "<EOS>", "<GO>"]

# Add codes to vocab
for code in codes:
    VOCAB_TO_INT[code] = len(VOCAB_TO_INT)

# Dictionary to convert integers to words
int_to_vocab = {}
for word, value in VOCAB_TO_INT.items():
    int_to_vocab[value] = word

usage_ratio = round(len(VOCAB_TO_INT) / len(WORD_COUNTS), 4) * 100

print("Total number of unique words:", len(WORD_COUNTS))
print("Number of words we will use:", len(VOCAB_TO_INT))
print("Percent of words we will use: {}%".format(usage_ratio))

# Need to use 300 for embedding dimensions to match CN's vectors.
embedding_dim = 300
nb_words = len(VOCAB_TO_INT)

# Create matrix with default values of zero
word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)
for word, i in VOCAB_TO_INT.items():
    if word in EMBEDDING_INDEX:
        word_embedding_matrix[i] = EMBEDDING_INDEX[word]
    else:
        # If word not in CN, create a random embedding for it
        new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
        EMBEDDING_INDEX[word] = new_embedding
        word_embedding_matrix[i] = new_embedding

# Check if value matches len(VOCAB_TO_INT)
print(len(word_embedding_matrix))


def convert_to_ints(text, word_count, unk_count, eos=False):
    '''Convert words in text to an integer.
       If word is not in vocab_to_int, use UNK's integer.
       Total the number of words and UNKs.
       Add EOS token to the end of texts'''
    ints = []
    for sentence in text:
        sentence_ints = []
        for word in sentence.split():
            word_count += 1
            if word in VOCAB_TO_INT:
                sentence_ints.append(VOCAB_TO_INT[word])
            else:
                sentence_ints.append(VOCAB_TO_INT["<UNK>"])
                unk_count += 1
        if eos:
            sentence_ints.append(VOCAB_TO_INT["<EOS>"])
        ints.append(sentence_ints)
    return ints, word_count, unk_count


# Apply convert_to_ints to clean_summaries and clean_texts
word_count = 0
unk_count = 0

int_summaries, word_count, unk_count = convert_to_ints(CLEAN_SUMMARIES, word_count, unk_count)
int_texts, word_count, unk_count = convert_to_ints(CLEAN_TEXTS, word_count, unk_count, eos=True)

unk_percent = round(unk_count / word_count, 4) * 100

print("Total number of words in headlines:", word_count)
print("Total number of UNKs in headlines:", unk_count)
print("Percent of words that are UNK: {}%".format(unk_percent))


def create_lengths(text):
    '''Create a data frame of the sentence lengths from a text'''
    lengths = []
    for sentence in text:
        lengths.append(len(sentence))
    return pd.DataFrame(lengths, columns=['counts'])


lengths_summaries = create_lengths(int_summaries)
lengths_texts = create_lengths(int_texts)

print("Summaries:")
print(lengths_summaries.describe())
print()
print("Texts:")
print(lengths_texts.describe())

# In[18]:


# Inspect the length of texts
print(np.percentile(lengths_texts.counts, 90))
print(np.percentile(lengths_texts.counts, 95))
print(np.percentile(lengths_texts.counts, 99))

# Inspect the length of summaries
print(np.percentile(lengths_summaries.counts, 90))
print(np.percentile(lengths_summaries.counts, 95))


def unk_counter(sentence):
    """Counts the number of time UNK appears in a sentence."""
    unk_count = 0
    for word in sentence:
        if word == VOCAB_TO_INT["<UNK>"]:
            unk_count += 1
    return unk_count


# Sort the summaries and texts by the length of the texts, shortest to longest
# Limit the length of summaries and texts based on the min and max ranges.
# Remove reviews that include too many UNKs

sorted_summaries = []
sorted_texts = []
max_text_length = 84
MAX_SUMMARY_LENGTH = 13
min_length = 2
unk_text_limit = 1
unk_summary_limit = 0

for length in range(min(lengths_texts.counts), max_text_length):
    for count, words in enumerate(int_summaries):
        if (min_length <= len(int_summaries[count]) <= MAX_SUMMARY_LENGTH and
                len(int_texts[count]) >= min_length and
                unk_counter(int_summaries[count]) <= unk_summary_limit and
                unk_counter(int_texts[count]) <= unk_text_limit and
                length == len(int_texts[count])):
            sorted_summaries.append(int_summaries[count])
            sorted_texts.append(int_texts[count])

# Compare lengths to ensure they match
print(len(sorted_summaries))
print(len(sorted_texts))


def model_inputs():
    """Create palceholders for inputs to the model"""

    INPUT_DATA = tf.placeholder(tf.int32, [None, None], name='input')
    TARGETS = tf.placeholder(tf.int32, [None, None], name='TARGETS')
    LR = tf.placeholder(tf.float32, name='LEARNING_RATE')
    KEEP_PROB = tf.placeholder(tf.float32, name='KEEP_PROB')
    SUMMARY_LENGTH = tf.placeholder(tf.int32, (None,), name='SUMMARY_LENGTH')
    MAX_SUMMARY_LENGTH = tf.reduce_max(SUMMARY_LENGTH, name='max_dec_len')
    TEXT_LENGTH = tf.placeholder(tf.int32, (None,), name='TEXT_LENGTH')

    return INPUT_DATA, TARGETS, LR, KEEP_PROB, SUMMARY_LENGTH, MAX_SUMMARY_LENGTH, TEXT_LENGTH


def process_encoding_input(target_data, VOCAB_TO_INT, BATCH_SIZE):
    """Remove the last word id from each batch and concat the <GO> to the begining of each batch"""

    ending = tf.strided_slice(target_data, [0, 0], [BATCH_SIZE, -1], [1, 1])
    dec_input = tf.concat([tf.fill([BATCH_SIZE, 1], VOCAB_TO_INT['<GO>']), ending], 1)

    return dec_input


def encoding_layer(RNN_SIZE, sequence_length, NUM_LAYERS, rnn_inputs, KEEP_PROB):
    """Create the encoding layer"""

    for layer in range(NUM_LAYERS):
        with tf.variable_scope('encoder_{}'.format(layer)):
            cell_fw = tf.contrib.rnn.LSTMCell(RNN_SIZE,
                                              initializer=tf.random_uniform_initializer(
                                                  -0.1, 0.1, seed=2))
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                                    input_keep_prob=KEEP_PROB)

            cell_bw = tf.contrib.rnn.LSTMCell(RNN_SIZE,
                                              initializer=tf.random_uniform_initializer(
                                                  -0.1, 0.1, seed=2))
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,
                                                    input_keep_prob=KEEP_PROB)

            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                    cell_bw,
                                                                    rnn_inputs,
                                                                    sequence_length,
                                                                    dtype=tf.float32)
    # Join outputs since we are using a bidirectional RNN
    enc_output = tf.concat(enc_output, 2)

    return enc_output, enc_state


def training_decoding_layer(dec_embed_input, SUMMARY_LENGTH, dec_cell, initial_state, output_layer,
                            vocab_size, MAX_SUMMARY_LENGTH):
    """Create the training logits"""

    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                        sequence_length=SUMMARY_LENGTH,
                                                        time_major=False)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                       training_helper,
                                                       initial_state,
                                                       output_layer)

    TRAINING_LOGITS, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                           output_time_major=False,
                                                           impute_finished=True,
                                                           maximum_iterations=MAX_SUMMARY_LENGTH)
    return TRAINING_LOGITS


def inference_decoding_layer(embeddings, start_token, end_token,
                             dec_cell, initial_state, output_layer,
                             MAX_SUMMARY_LENGTH, BATCH_SIZE):
    """Create the inference logits"""

    start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32),
                           [BATCH_SIZE], name='start_tokens')

    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                start_tokens,
                                                                end_token)

    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                        inference_helper,
                                                        initial_state,
                                                        output_layer)

    INFERENCE_LOGITS, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                            output_time_major=False,
                                                            impute_finished=True,
                                                            maximum_iterations=MAX_SUMMARY_LENGTH)

    return INFERENCE_LOGITS


def decoding_layer(dec_embed_input, embeddings, enc_output,
                   enc_state, vocab_size, TEXT_LENGTH,
                   SUMMARY_LENGTH,
                   MAX_SUMMARY_LENGTH, RNN_SIZE, VOCAB_TO_INT,
                   KEEP_PROB, BATCH_SIZE, NUM_LAYERS):
    """Create the decoding cell and attention for the training and inference decoding layers"""

    for layer in range(NUM_LAYERS):
        with tf.variable_scope('decoder_{}'.format(layer)):
            lstm = tf.contrib.rnn.LSTMCell(RNN_SIZE,
                                           initializer=tf.random_uniform_initializer(
                                               -0.1, 0.1, seed=2))
            dec_cell = tf.contrib.rnn.DropoutWrapper(lstm,
                                                     input_keep_prob=KEEP_PROB)

    output_layer = Dense(vocab_size,
                         kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    attn_mech = tf.contrib.seq2seq.BahdanauAttention(RNN_SIZE,
                                                     enc_output,
                                                     TEXT_LENGTH,
                                                     normalize=False,
                                                     name='BahdanauAttention')

    dec_cell = tf.contrib.seq2seq.DynamicAttentionWrapper(dec_cell,
                                                          attn_mech,
                                                          RNN_SIZE)

    initial_state = tf.contrib.seq2seq.DynamicAttentionWrapperState(enc_state[0],
                                                                    _zero_state_tensors(RNN_SIZE,
                                                                                        BATCH_SIZE,
                                                                                        tf.float32))
    with tf.variable_scope("decode"):
        TRAINING_LOGITS = training_decoding_layer(dec_embed_input,
                                                  SUMMARY_LENGTH,
                                                  dec_cell,
                                                  initial_state,
                                                  output_layer,
                                                  vocab_size,
                                                  MAX_SUMMARY_LENGTH)
    with tf.variable_scope("decode", reuse=True):
        INFERENCE_LOGITS = inference_decoding_layer(embeddings,
                                                    VOCAB_TO_INT['<GO>'],
                                                    VOCAB_TO_INT['<EOS>'],
                                                    dec_cell,
                                                    initial_state,
                                                    output_layer,
                                                    MAX_SUMMARY_LENGTH,
                                                    BATCH_SIZE)

    return TRAINING_LOGITS, INFERENCE_LOGITS


def seq2seq_model(INPUT_DATA, target_data, KEEP_PROB,
                  TEXT_LENGTH, SUMMARY_LENGTH, MAX_SUMMARY_LENGTH,
                  vocab_size, RNN_SIZE, NUM_LAYERS,
                  VOCAB_TO_INT, BATCH_SIZE):
    """Use the previous functions to create the training and inference logits"""

    # Use Numberbatch's embeddings and the newly created ones as our embeddings
    embeddings = word_embedding_matrix

    enc_embed_input = tf.nn.embedding_lookup(embeddings, INPUT_DATA)
    enc_output, enc_state = encoding_layer(RNN_SIZE,
                                           TEXT_LENGTH, NUM_LAYERS,
                                           enc_embed_input, KEEP_PROB)

    dec_input = process_encoding_input(target_data, VOCAB_TO_INT, BATCH_SIZE)
    dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)

    TRAINING_LOGITS, INFERENCE_LOGITS = decoding_layer(dec_embed_input,
                                                       embeddings,
                                                       enc_output,
                                                       enc_state,
                                                       vocab_size,
                                                       TEXT_LENGTH,
                                                       SUMMARY_LENGTH,
                                                       MAX_SUMMARY_LENGTH,
                                                       RNN_SIZE,
                                                       VOCAB_TO_INT,
                                                       KEEP_PROB,
                                                       BATCH_SIZE,
                                                       NUM_LAYERS)

    return TRAINING_LOGITS, INFERENCE_LOGITS


def pad_sentence_batch(sentence_batch):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [VOCAB_TO_INT['<PAD>']] * (max_sentence - len(sentence))
            for sentence in sentence_batch]


def get_batches(summaries, texts, BATCH_SIZE):
    """Batch summaries, texts, and the lengths of their sentences together"""
    for batch_i in range(0, len(texts) // BATCH_SIZE):
        start_i = batch_i * BATCH_SIZE
        summaries_batch = summaries[start_i:start_i + BATCH_SIZE]
        texts_batch = texts[start_i:start_i + BATCH_SIZE]
        pad_summaries_batch = np.array(pad_sentence_batch(summaries_batch))
        pad_texts_batch = np.array(pad_sentence_batch(texts_batch))

        # Need the lengths for the _lengths parameters
        pad_summaries_lengths = []
        for summary in pad_summaries_batch:
            pad_summaries_lengths.append(len(summary))

        pad_texts_lengths = []
        for text in pad_texts_batch:
            pad_texts_lengths.append(len(text))

        yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths


# Set the Hyperparameters
EPOCHS = 5
BATCH_SIZE = 64
RNN_SIZE = 256
NUM_LAYERS = 2
LEARNING_RATE = 0.005
KEEP_PROBABILITY = 0.75

# Build the graph
TRAIN_GRAPH = tf.Graph()
# Set the graph to default to ensure that it is ready for training
with TRAIN_GRAPH.as_default():
    # Load the model inputs
    INPUT_DATA, TARGETS, LR, KEEP_PROB, SUMMARY_LENGTH, \
    MAX_SUMMARY_LENGTH, TEXT_LENGTH = model_inputs()

    # Create the training and inference logits
    TRAINING_LOGITS, INFERENCE_LOGITS = seq2seq_model(tf.reverse(INPUT_DATA, [-1]),
                                                      TARGETS,
                                                      KEEP_PROB,
                                                      TEXT_LENGTH,
                                                      SUMMARY_LENGTH,
                                                      MAX_SUMMARY_LENGTH,
                                                      len(VOCAB_TO_INT) + 1,
                                                      RNN_SIZE,
                                                      NUM_LAYERS,
                                                      VOCAB_TO_INT,
                                                      BATCH_SIZE)

    # Create tensors for the training logits and inference logits
    TRAINING_LOGITS = tf.identity(TRAINING_LOGITS.rnn_output, 'logits')
    INFERENCE_LOGITS = tf.identity(INFERENCE_LOGITS.sample_id, name='predictions')

    # Create the weights for sequence_loss
    MASKS = tf.sequence_mask(SUMMARY_LENGTH, MAX_SUMMARY_LENGTH, dtype=tf.float32, name='MASKS')

    with tf.name_scope("optimization"):
        # Loss function
        COST = tf.contrib.seq2seq.sequence_loss(
            TRAINING_LOGITS,
            TARGETS,
            MASKS)

        # OPTIMIZER
        OPTIMIZER = tf.train.AdamOptimizer(LEARNING_RATE)

        # Gradient Clipping
        GRADIENTS = OPTIMIZER.compute_gradients(COST)
        CAPPED_GRADIENTS = [(tf.clip_by_value(grad, -5., 5.),
                             var)
                            for grad, var in GRADIENTS if grad is not None]
        TRAIN_OP = OPTIMIZER.apply_gradients(CAPPED_GRADIENTS)
print("Graph is built.")

# Subset the data for training
START = 400000
END = START + 50000
SORTED_SUMMARIES_SHORT = sorted_summaries[START:END]
SORTED_TEXTS_SHORT = sorted_texts[START:END]
print("The shortest text length:", len(SORTED_TEXTS_SHORT[0]))
print("The longest text length:", len(SORTED_TEXTS_SHORT[-1]))

# Train the Model
LEARNING_RATE_DECAY = 0.95
MIN_LEARNING_RATE = 0.0005
DISPLAY_STEP = 20  # Check training loss after every 20 batches
STOP_EARLY = 0
STOP = 3  # If the update loss does not decrease in 3 consecutive update checks, stop training
PER_EPOCH = 3  # Make 3 update checks per epoch
UPDATE_CHECK = (len(SORTED_TEXTS_SHORT) // BATCH_SIZE // PER_EPOCH) - 1

UPDATE_LOSS = 0
BATCH_LOSS = 0
SUMMARY_UPDATE_LOSS = []  # Record the update losses for saving improvements in the model

CHECKPOINT = "best_model.ckpt"
with tf.Session(graph=TRAIN_GRAPH) as sess:
    sess.run(tf.global_variables_initializer())

    # If we want to continue training a previous session
    # loader = tf.train.import_meta_graph("./" + checkpoint + '.meta')
    # loader.restore(sess, checkpoint)

    for epoch_i in range(1, EPOCHS + 1):
        UPDATE_LOSS = 0
        BATCH_LOSS = 0
        for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(
                get_batches(SORTED_SUMMARIES_SHORT, SORTED_TEXTS_SHORT, BATCH_SIZE)):
            start_time = time.time()
            _, loss = sess.run(
                [TRAIN_OP, COST],
                {INPUT_DATA: texts_batch,
                 TARGETS: summaries_batch,
                 LR: LEARNING_RATE,
                 SUMMARY_LENGTH: summaries_lengths,
                 TEXT_LENGTH: texts_lengths,
                 KEEP_PROB: KEEP_PROBABILITY})

            BATCH_LOSS += loss
            UPDATE_LOSS += loss
            end_time = time.time()
            batch_time = end_time - start_time

            if batch_i % DISPLAY_STEP == 0 and batch_i > 0:
                print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                      .format(epoch_i,
                              EPOCHS,
                              batch_i,
                              len(SORTED_TEXTS_SHORT) // BATCH_SIZE,
                              BATCH_LOSS / DISPLAY_STEP,
                              batch_time * DISPLAY_STEP))
                BATCH_LOSS = 0

            if batch_i % UPDATE_CHECK == 0 and batch_i > 0:
                print("Average loss for this update:", round(UPDATE_LOSS / UPDATE_CHECK, 3))
                SUMMARY_UPDATE_LOSS.append(UPDATE_LOSS)

                # If the update loss is at a new minimum, save the model
                if UPDATE_LOSS <= min(SUMMARY_UPDATE_LOSS):
                    print('New Record!')
                    STOP_EARLY = 0
                    saver = tf.train.Saver()
                    saver.save(sess, CHECKPOINT)

                else:
                    print("No Improvement.")
                    STOP_EARLY += 1
                    if STOP_EARLY == STOP:
                        break
                UPDATE_LOSS = 0

        # Reduce learning rate, but not below its minimum value
        LEARNING_RATE *= LEARNING_RATE_DECAY
        if LEARNING_RATE < MIN_LEARNING_RATE:
            LEARNING_RATE = MIN_LEARNING_RATE

        if STOP_EARLY == STOP:
            print("STOPping Training.")
            break


def text_to_seq(text):
    '''Prepare the text for the model'''

    text = clean_text(text)
    return [VOCAB_TO_INT.get(word, VOCAB_TO_INT['<UNK>']) for word in text.split()]


# Create your own review or use one from the dataset
# input_sentence = "I have never eaten an apple before, but this red one was nice. \
# I think that I will try a green apple next time."
# text = text_to_seq(input_sentence)
RANDOM = np.random.randint(0, len(CLEAN_TEXTS))
INPUT_SENTENCE = CLEAN_TEXTS[RANDOM]
TEXT = text_to_seq(CLEAN_TEXTS[RANDOM])

CHECKPOINT = "./best_model.ckpt"

LOADED_GRAPH = tf.Graph()
with tf.Session(graph=LOADED_GRAPH) as sess:
    # Load saved model
    LOADER = tf.train.import_meta_graph(CHECKPOINT + '.meta')
    LOADER.restore(sess, CHECKPOINT)

    INPUT_DATA = LOADED_GRAPH.get_tensor_by_name('input:0')
    logits = LOADED_GRAPH.get_tensor_by_name('predictions:0')
    TEXT_LENGTH = LOADED_GRAPH.get_tensor_by_name('TEXT_LENGTH:0')
    SUMMARY_LENGTH = LOADED_GRAPH.get_tensor_by_name('SUMMARY_LENGTH:0')
    KEEP_PROB = LOADED_GRAPH.get_tensor_by_name('KEEP_PROB:0')

    # Multiply by batch_size to match the model's input parameters
    ANSWER_LOGITS = sess.run(logits, {INPUT_DATA: [TEXT] * BATCH_SIZE,
                                      SUMMARY_LENGTH: [np.random.randint(5, 8)],
                                      TEXT_LENGTH: [len(TEXT)] * BATCH_SIZE,
                                      KEEP_PROB: 1.0})[0]

# Remove the PADding from the tweet
PAD = VOCAB_TO_INT["<PAD>"]

print('Original Text:', INPUT_SENTENCE)

print('\nText')
print('  Word Ids:    {}'.format([i for i in TEXT]))
print('  Input Words: {}'.format(" ".join([int_to_vocab[i] for i in text])))

print('\nSummary')
print('  Word Ids:       {}'.format([i for i in ANSWER_LOGITS if i != PAD]))
print('  Response Words: {}'.format(" ".join([int_to_vocab[i] for i in ANSWER_LOGITS if i != PAD])))
