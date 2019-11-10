#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 02:07:17 2019

@author: lumi
"""

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import tensorflow as tf
import re
from nltk.corpus import stopwords
import time
from tensorflow.python.layers.core import Dense

# from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors

# from attention import AttentionLayer

# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences


DATA = pd.read_csv('posts_content.csv')

DATA.shape

# check for null values
DATA.isnull().sum()
# remove null values and unwanted features
DATA = DATA.dropna()
DATA = DATA.drop(['user_id', 'tags', 'Unnamed: 4'], axis=1)
DATA = DATA.reset_index(drop=True)

DATA.head()

DATA.info()

# prepare the data
CONTRACTION_MAPPING = {"ain't": "is not", "aren't": "are not",
                       "can't": "cannot", "'cause": "because",
                       "could've": "could have", "couldn't": "could not",
                       "didn't": "did not", "doesn't": "does not",
                       "don't": "do not", "hadn't": "had not",
                       "hasn't": "has not", "haven't": "have not",
                       "he'd": "he would", "he'll": "he will",
                       "he's": "he is", "how'd": "how did",
                       "how'd'y": "how do you", "how'll": "how will",
                       "how's": "how is",
                       "I'd": "I would", "I'd've": "I would have",
                       "I'll": "I will", "I'll've": "I will have",
                       "I'm": "I am", "I've": "I have", "i'd": "i would",
                       "i'd've": "i would have", "i'll": "i will",
                       "i'll've": "i will have", "i'm": "i am",
                       "i've": "i have", "isn't": "is not", "it'd": "it would",
                       "it'd've": "it would have", "it'll": "it will",
                       "it'll've": "it will have", "it's": "it is",
                       "let's": "let us", "ma'am": "madam",
                       "mayn't": "may not", "might've": "might have",
                       "mightn't": "might not",
                       "mightn't've": "might not have", "must've": "must have",
                       "mustn't": "must not", "mustn't've": "must not have",
                       "needn't": "need not",
                       "needn't've": "need not have", "o'clock": "of the clock",
                       "oughtn't": "ought not", "oughtn't've": "ought not have",
                       "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                       "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                       "she'll've": "she will have", "she's": "she is",
                       "should've": "should have", "shouldn't": "should not",
                       "shouldn't've": "should not have",
                       "so've": "so have", "so's": "so as",
                       "this's": "this is", "that'd": "that would", "that'd've": "that would have",
                       "that's": "that is",
                       "there'd": "there would",
                       "there'd've": "there would have", "there's": "there is", "here's": "here is",
                       "they'd": "they would", "they'd've": "they would have",
                       "they'll": "they will", "they'll've": "they will have",
                       "they're": "they are", "they've": "they have", "to've": "to have",
                       "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
                       "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                       "we've": "we have", "weren't": "were not", "what'll": "what will",
                       "what'll've": "what will have", "what're": "what are",
                       "what's": "what is", "what've": "what have", "when's": "when is",
                       "when've": "when have",
                       "where'd": "where did", "where's": "where is",
                       "where've": "where have", "who'll": "who will",
                       "who'll've": "who will have", "who's": "who is",
                       "who've": "who have",
                       "why's": "why is", "why've": "why have", "will've": "will have",
                       "won't": "will not",
                       "won't've": "will not have",
                       "would've": "would have", "wouldn't": "would not",
                       "wouldn't've": "would not have",
                       "y'all": "you all",
                       "y'all'd": "you all would", "y'all'd've": "you all would have",
                       "y'all're": "you all are",
                       "y'all've": "you all have",
                       "you'd": "you would", "you'd've": "you would have",
                       "you'll": "you will",
                       "you'll've": "you will have",
                       "you're": "you are", "you've": "you have"}


def text_cleaner(TEXT, remove_stopwords=True):
    '''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings'''
    # convert to text to lower text
    new_string = TEXT.lower()
    new_string = BeautifulSoup(new_string, "lxml").TEXT
    # replace contraction with their long form
    if True:
        new_string = new_string.split()
        new_TEXT = []
        for word in TEXT:
            if word in CONTRACTION_MAPPING:
                new_text.append(CONTRACTION_MAPPING[word])
            else:
                new_text.append(word)
        new_string = " ".join(new_text)

    # Format words and remove unwanted characters
    new_string = re.sub(r'https?:\/\/.*[\r\n]*', '', new_string, flags=re.MULTILINE)
    new_string = re.sub(r'\<a href', ' ', new_string)
    new_string = re.sub(r'&amp;', '', new_string)
    new_string = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', new_string)
    new_string = re.sub(r'<br />', ' ', new_string)
    new_string = re.sub(r'\'', ' ', new_string)

    # remove stopwords
    if remove_stopwords:
        new_string = new_string.split()
        stops = set(stopwords.words("english"))
        new_string = [w for w in new_string if not w in stops]
        new_string = " ".join(new_string)

    return new_string


# clean the summaries and texts

CLEANED_TEXT = []
for t in DATA['content']:
    CLEANED_TEXT.append(text_cleaner(t))

CLEANED_SUMMARY = []
for t in DATA['title']:
    CLEANED_SUMMARY.append(text_cleaner(t, remove_stopwords=False))

DATA['CLEANED_TEXT'] = CLEANED_TEXT
DATA['CLEANED_SUMMARY'] = CLEANED_SUMMARY


def count_words(count_dict, TEXT):
    '''Count the number of occurrences of each word in a set of TEXT'''
    for sentence in TEXT:
        for word in sentence.split():
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1


# find the number of times a word appears in the TEXT and the size of the vocabulary

word_counts = {}

count_words(word_counts, CLEANED_SUMMARY)
count_words(word_counts, CLEANED_TEXT)

Size_of_Vocabulary = len(word_counts)

# Load Conceptnet Numberbatch's (CN) embeddings, similar to GloVe, but probably better
# (https://github.com/commonsense/conceptnet-numberbatch)
embeddings_index = {}
with open('https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-19.08.txt.gz',
          encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding

Word_embeddings = len(embeddings_index)

# Find the number of words that are missing from CN, and are used more than our threshold.
missing_words = 0
threshold = 20

for word, count in word_counts.items():
    if count > threshold:
        if word not in embeddings_index:
            missing_words += 1

missing_ratio = round(missing_words / len(word_counts), 4) * 100

# print("Number of words missing from CN:", missing_words)
# print("Percent of words that are missing from vocabulary: {}%".format(missing_ratio))


# Limit the vocab that we will use to words that appear ≥ threshold or are in GloVe

# dictionary to convert words to integers
vocab_to_int = {}

value = 0
for word, count in word_counts.items():
    if count >= threshold or word in embeddings_index:
        vocab_to_int[word] = value
        value += 1

# Special tokens that will be added to our vocab
codes = ["<UNK>", "<PAD>", "<EOS>", "<GO>"]

# Add codes to vocab
for code in codes:
    vocab_to_int[code] = len(vocab_to_int)

# Dictionary to convert integers to words
int_to_vocab = {}
for word, value in vocab_to_int.items():
    int_to_vocab[value] = word

usage_ratio = round(len(vocab_to_int) / len(word_counts), 4) * 100

len_word_counts = len(word_counts)
len_vocab_to_int = len(vocab_to_int)

# Need to use 300 for embedding dimensions to match CN's vectors.
embedding_dim = 300
nb_words = len(vocab_to_int)

# Create matrix with default values of zero
word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)
for word, i in vocab_to_int.items():
    if word in embeddings_index:
        word_embedding_matrix[i] = embeddings_index[word]
    else:
        # If word not in CN, create a random embedding for it
        new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
        embeddings_index[word] = new_embedding
        word_embedding_matrix[i] = new_embedding

# Check if value matches len(vocab_to_int)
len_embedding_matrix = len(word_embedding_matrix)


def convert_to_ints(TEXT, word_count, unk_count, eos=False):
    '''Convert words in TEXT to an integer.
       If word is not in vocab_to_int, use UNK's integer.
       Total the number of words and UNKs.
       Add EOS token to the end of TEXTs'''
    ints = []
    for sentence in TEXT:
        sentence_ints = []
        for word in sentence.split():
            word_count += 1
            if word in vocab_to_int:
                sentence_ints.append(vocab_to_int[word])
            else:
                sentence_ints.append(vocab_to_int["<UNK>"])
                unk_count += 1
        if eos:
            sentence_ints.append(vocab_to_int["<EOS>"])
        ints.append(sentence_ints)
    return ints, word_count, unk_count


# Apply convert_to_ints to clean_summaries and clean_TEXTs
word_count = 0
unk_count = 0

int_summaries, word_count, unk_count = convert_to_ints(CLEANED_SUMMARY, word_count, unk_count)
int_texts, word_count, unk_count = convert_to_ints(CLEANED_TEXT, word_count, unk_count, eos=True)

unk_percent = round(unk_count / word_count, 4) * 100

words_in_headlines = word_count
UNKs_in_headlines = unk_count
Percent_words_UNK = unk_percent


def create_lengths(TEXT):
    '''Create a data frame of the sentence lengths from a TEXT'''
    lengths = []
    for sentence in TEXT:
        lengths.append(len(sentence))
    return pd.DataFrame(lengths, columns=['counts'])


lengths_summaries = create_lengths(int_summaries)
lengths_texts = create_lengths(int_texts)

lengths_summaries.describe()
lengths_texts.describe()


def unk_counter(sentence):
    '''Counts the number of time UNK appears in a sentence.'''
    unk_count = 0
    for word in sentence:
        if word == vocab_to_int["<UNK>"]:
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
        if (len(int_summaries[count]) >= min_length and
                len(int_summaries[count]) <= MAX_SUMMARY_LENGTH and
                len(int_texts[count]) >= min_length and
                unk_counter(int_summaries[count]) <= unk_summary_limit and
                unk_counter(int_texts[count]) <= unk_text_limit and
                length == len(int_texts[count])
        ):
            sorted_summaries.append(int_summaries[count])
            sorted_texts.append(int_texts[count])

# Compare lengths to ensure they match
len_sorted_summaries = len(sorted_summaries)
len_sorted_texts = len(sorted_texts)


# building the model
def model_inputs():
    '''Create palceholders for inputs to the model'''

    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    TARGETS = tf.placeholder(tf.int32, [None, None], name='TARGETS')
    LR = tf.placeholder(tf.float32, name='learning_rate')
    KEEP_PROB = tf.placeholder(tf.float32, name='KEEP_PROB')
    SUMMARY_LENGTH = tf.placeholder(tf.int32, (None,), name='SUMMARY_LENGTH')
    MAX_SUMMARY_LENGTH = tf.reduce_max(SUMMARY_LENGTH, name='max_dec_len')
    text_length = tf.placeholder(tf.int32, (None,), name='text_length')

    return input_data, TARGETS, LR, KEEP_PROB, SUMMARY_LENGTH, MAX_SUMMARY_LENGTH, TEXT_LENGTH


def process_encoding_input(target_data, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)
    return dec_input


def encoding_layer(RNN_SIZE, sequence_length, NUM_LAYERS, rnn_inputs, KEEP_PROB):
    '''Create the encoding layer'''
    for layer in range(NUM_LAYERS):
        with tf.variable_scope('encoder_{}'.format(layer)):
            cell_fw = tf.contrib.rnn.LSTMCell(RNN_SIZE,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                                    input_keep_prob=KEEP_PROB)
            cell_bw = tf.contrib.rnn.LSTMCell(RNN_SIZE,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,
                                                    input_keep_prob=KEEP_PROB)
            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                    cell_bw,
                                                                    rnn_inputs,
                                                                    sequence_length,
                                                                    dtype=tf.float32)
            # Join outputs since we are using a bidirectional RNN\n",
    enc_output = tf.concat(enc_output, 2)

    return enc_output, enc_state


def training_decoding_layer(dec_embed_input, SUMMARY_LENGTH, dec_cell, initial_state, output_layer,
                            vocab_size, MAX_SUMMARY_LENGTH):
    '''Create the training logits'''

    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                        sequence_length=SUMMARY_LENGTH,
                                                        time_major=False)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                       training_helper,
                                                       initial_state,
                                                       output_layer)

    TRAINING_LOGITS, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                              output_time_major=False,
                                                              impute_finished=True,
                                                              maximum_iterations=MAX_SUMMARY_LENGTH)
    return training_decoder


def inference_decoding_layer(embeddings, START_TOKEN, end_token, dec_cell, initial_state, output_layer,
                             MAX_SUMMARY_LENGTH, batch_size):
    '''Create the inference logits'''

    START_TOKENS = tf.tile(tf.constant([START_TOKEN], dtype=tf.int32), [batch_size], name='START_TOKENS')

    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                START_TOKENS,
                                                                end_token)
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                        inference_helper,
                                                        initial_state,
                                                        output_layer)
    INFERENCE_LOGITS, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                               output_time_major=False,
                                                               impute_finished=True,
                                                               maximum_iterations=MAX_SUMMARY_LENGTH)
    return inference_decoder


def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, TEXT_LENGTH, SUMMARY_LENGTH,
                   MAX_SUMMARY_LENGTH, RNN_SIZE, vocab_to_int, KEEP_PROB, batch_size, NUM_LAYERS):
    '''Create the decoding cell and attention for the training and inference decoding layers'''

    for layer in range(NUM_LAYERS):
        with tf.variable_scope('decoder_{}'.format(layer)):
            lstm = tf.contrib.rnn.LSTMCell(RNN_SIZE,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            dec_cell = tf.contrib.rnn.DropoutWrapper(lstm,
                                                     input_keep_prob=KEEP_PROB)
    output_layer = Dense(vocab_size,
                         kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    attn_mech = tf.contrib.seq2seq.BahdanauAttention(RNN_SIZE,
                                                     enc_output,
                                                     TEXT_LENGTH,
                                                     normalize=False,
                                                     name='BahdanauAttention')
    dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell,
                                                   attn_mech,
                                                   RNN_SIZE)

    # initial_state = tf.contrib.seq2seq.AttentionWrapperState(enc_state[0],
    #                                                               _zero_state_tensors(RNN_SIZE,
    #                                                                                   batch_size, 
    #                                                                                   tf.float32)) 
    initial_state = dec_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=enc_state[0])

    with tf.variable_scope("decode"):
        training_decoder = training_decoding_layer(dec_embed_input,
                                                   summary_length,
                                                   dec_cell,
                                                   initial_state,
                                                   output_layer,
                                                   vocab_size,
                                                   MAX_SUMMARY_LENGTH)
        TRAINING_LOGITS, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                  output_time_major=False,
                                                                  impute_finished=True,
                                                                  maximum_iterations=MAX_SUMMARY_LENGTH)

    with tf.variable_scope("decode", reuse=True):
        inference_decoder = inference_decoding_layer(embeddings,
                                                     vocab_to_int['<GO>'],
                                                     vocab_to_int['<EOS>'],
                                                     dec_cell,
                                                     initial_state,
                                                     output_layer,
                                                     MAX_SUMMARY_LENGTH,
                                                     batch_size)
        INFERENCE_LOGITS, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                   output_time_major=False,
                                                                   impute_finished=True,
                                                                   maximum_iterations=MAX_SUMMARY_LENGTH)

    return TRAINING_LOGITS, INFERENCE_LOGITS


def seq2seq_model(input_data, target_data, KEEP_PROB, TEXT_LENGTH, SUMMARY_LENGTH, MAX_SUMMARY_LENGTH,
                  vocab_size, RNN_SIZE, NUM_LAYERS, vocab_to_int, batch_size):
    '''Use the previous functions to create the training and inference logits'''
    # Use Numberbatch's embeddings and the newly created ones as our embeddings
    embeddings = word_embedding_matrix

    enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)

    enc_output, enc_state = encoding_layer(RNN_SIZE, TEXT_LENGTH, NUM_LAYERS, enc_embed_input, KEEP_PROB)

    dec_input = process_encoding_input(target_data, vocab_to_int, batch_size)

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
                                                       vocab_to_int,
                                                       KEEP_PROB,
                                                       batch_size,
                                                       NUM_LAYERS)
    return TRAINING_LOGITS, INFERENCE_LOGITS


def pad_sentence_batch(sentence_batch):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(summaries, texts, batch_size):
    """Batch summaries, texts, and the lengths of their sentences together"""
    for batch_i in range(0, len(texts) // batch_size):
        start_i = batch_i * batch_size
        summaries_batch = summaries[start_i:start_i + batch_size]
        texts_batch = texts[start_i:start_i + batch_size]
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
EPOCHS = 100
batch_size = 64
RNN_SIZE = 256
NUM_LAYERS = 2
learning_rate = 0.005
KEEP_PROBABILITY = 0.75

# Build the graph
TRAIN_GRAPH = tf.Graph()
# Set the graph to default to ensure that it is ready for training

with TRAIN_GRAPH.as_default():
    # Load the model inputs
    input_data, TARGETS, LR, KEEP_PROB, SUMMARY_LENGTH, MAX_SUMMARY_LENGTH, TEXT_LENGTH = model_inputs()
    # Create the training and inference logits
    TRAINING_LOGITS, INFERENCE_LOGITS = seq2seq_model(tf.reverse(input_data, [-1]),
                                                      TARGETS,
                                                      KEEP_PROB,
                                                      TEXT_LENGTH,
                                                      SUMMARY_LENGTH,
                                                      MAX_SUMMARY_LENGTH,
                                                      len(vocab_to_int) + 1,
                                                      RNN_SIZE,
                                                      NUM_LAYERS,
                                                      vocab_to_int,
                                                      batch_size)
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
        GRADIENTS = OPTIMIZER.compute_GRADIENTS(COST)
        CAPPED_GRADIENTS = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in GRADIENTS if grad is not None]
        TRAIN_OP = OPTIMIZER.apply_gradients(CAPPED_GRADIENTS)
print("Graph is built.")

# training the model
# the training of the model starts from  a subset inorder not to make it easy for the model
# The texts used are closer to the median lengths

# Subset the data for training
start = 200000
END = start + 50000
SORTED_SUMMARIES_SHORT = sorted_summaries[start:END]
SORTED_TEXTS_SHORT = sorted_texts[start:END]
print("The shortest text length:", len(SORTED_TEXTS_SHORT[0]))
print("The longest text length:", len(SORTED_TEXTS_SHORT[-1]))

# Train the Model
LEARNING_RATE_DECAY = 0.95
MIN_LEARNING_RATE = 0.0005
DISPLAY_STEP = 20  # Check training loss after every 20 batches
stop_early = 0
stop = 3  # If the update loss does not decrease in 3 consecutive update checks, stop training
PER_EPOCH = 3  # Make 3 update checks per epoch
UPDATE_CHECK = (len(SORTED_TEXTS_SHORT) // batch_size // PER_EPOCH) - 1

UPDATE_LOSS = 0
BATCH_LOSS = 0
SUMMARY_UPDATE_LOSS = []  # Record the update losses for saving improvements in the model

tf.reset_default_graph()
CHECKPOINT = "best_model.ckpt"
with tf.Session(graph=TRAIN_GRAPH) as sess:
    sess.run(tf.global_variables_initializer())
    # to continue training a previous session
    # loader = tf.train.import_meta_graph(CHECKPOINT + '.meta')
    # loader.restore(sess, CHECKPOINT)
    # sess.run(tf.local_variables_initializer())
    for epoch_i in range(1, EPOCHS + 1):
        UPDATE_LOSS = 0
        BATCH_LOSS = 0
        for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(
                get_batches(SORTED_SUMMARIES_SHORT, SORTED_TEXTS_SHORT, batch_size)):

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

                # saver = tf.train.Saver()
                # saver.save(sess, CHECKPOINT)
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
                    if STOP_EARLY == stop:
                        break
                UPDATE_LOSS = 0
        # Reduce learning rate, but not below its minimum value
        LEARNING_RATE *= LEARNING_RATE_DECAY
        if LEARNING_RATE < MIN_LEARNING_RATE:
            LEARNING_RATE = MIN_LEARNING_RATE
        if STOP_EARLY == STOP:
            print("STOPping Training.")
            break

CHECKPOINT = "./best_model.ckpt"
LOADED_GRAPH = tf.Graph()
with tf.Session(graph=LOADED_GRAPH) as sess:
    # Load saved model
    LOADER = tf.train.import_meta_graph(CHECKPOINT + '.meta')
    LOADER.restore(sess, CHECKPOINT)
    NAMES = []
    [NAMES.append(n.name) for n in LOADED_GRAPH.as_graph_def().node]
NAMES


# making summaries
# to see the quality of the summaries this model can generate,use a article
# the length of the summary can be set to a fixed value, or use a RANDOM value as below.

def text_to_seq(text):
    '''Prepare the text for the model'''
    text = text_cleaner(text)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in text.split()]


# use an article
# input_sentence = "I have never eaten an apple before, but this red one was nice. \
# I think that I will try a green apple next time."
# text = text_to_seq(input_sentence)
RANDOM = np.random.randint(0, len(CLEANED_TEXT))
INPUT_SENTENCE = CLEANED_TEXT[RANDOM]
text = text_to_seq(CLEANED_TEXT[RANDOM])

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

    # Multiply by BATCH_SIZE to match the model's input parameters
    answer_logits = sess.run(logits, {INPUT_DATA: [text] * BATCH_SIZE,
                                      SUMMARY_LENGTH: [np.random.randint(5, 8)],
                                      TEXT_LENGTH: [len(text)] * BATCH_SIZE,
                                      KEEP_PROB: 1.0})[0]

# Remove the padding from the article
pad = vocab_to_int["<PAD>"]

print('Original Text:', INPUT_SENTENCE)

print('\nText')
print('  Word Ids:    {}'.format([i for i in TEXT]))
print('  Input Words: {}'.format(" ".join([int_to_vocab[i] for i in TEXT])))

print('\nSummary')
print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format(" ".join([int_to_vocab[i] for i in answer_logits if i != pad])))
