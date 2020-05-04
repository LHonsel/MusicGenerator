import pandas as pd
import numpy as np
import pretty_midi
import music21
import re
from random import sample
from keras.utils import to_categorical

# Create vocab dictionaries
def create_dictionaries(text_data):
    '''Create two dictionaries: one from character to integer,
    and one from integer to character'''
    char_list = list(text_data)
    vocab = list(set(char_list))
    
    # Dictionary with character to integer
    vocab_dict = {i: j for i,j in enumerate(char_list)}
    
    # Dictionary with integer to character
    vocab_dict_rev = {j: i for i,j in enumerate(char_list)}
        
    return vocab_dict, vocab_dict_rev


def encoder(text_data, dictionary):
    '''Convert text data into a numeric list.'''
    character_nums = list(text_data)
    
    for i in range(len(character_nums)):
        character_nums[i] = dictionary[character_nums[i]]
        
    return character_nums


def decoder(binary_matrix, dictionary):
    '''Convert numeric list a text string.'''
    text_list = []
    
    for i in range(len(numeric_data)):
        text_list.append(dictionary[numeric_data[i]])
    
    return "".join(text_list)


def create_training(char_nums, num_samples, str_length):
    '''Create training dataset with x and y values from your numeric list.
    The x data is a list of numeric sequences, and the y data is those sequences
    shifted one character to the right.'''
    # Get starting indices of the random samples for your training batch
    start_indices = sample(char_nums[0:(len(char_nums)-str_length-1)], num_samples)
    
    # The x_values begin at the starting indices and are str_length characters long
    # The y_values begin one character into the x_values and end one character longer than x_values
    x_data = np.array(char_nums[0:str_length])
    y_data = np.array(char_nums[1:str_length+1])
    for i in range(1,num_samples):
        x_data = np.vstack((x_data, np.array(char_nums[i:i+str_length])))
        y_data = np.vstack((y_data, np.array(char_nums[i+1:i+str_length+1])))
    
    #return x_data, y_data
    return x_data, y_data
