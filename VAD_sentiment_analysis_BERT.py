# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 16:00:47 2018

@author: Ryutaro Takanami
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import warnings
warnings.filterwarnings('ignore')

os.chdir(r"C:\Users\Ryutaro Takanami\.spyder-py3")

df_ang = pd.read_table('DissertationData\WASSA2017anger.txt', delimiter='\t', header=None)
df_fear = pd.read_table('DissertationData\WASSA2017fear.txt', delimiter='\t', header=None)
df_joy = pd.read_table('DissertationData\WASSA2017joy.txt', delimiter='\t', header=None)
df_sadness = pd.read_table('DissertationData\WASSA2017sadness.txt', delimiter='\t', header=None)
df_emobank = pd.read_table('DissertationData\emobank.txt', delimiter=',')
#df_SemEval2007test = pd.read_table('DissertationData\SemEval2007test.txt')
#df_SemEval2007train = pd.read_table('DissertationData\SemEval2007train.txt', error_bad_lines=False)
df_SSECtest = pd.read_table('DissertationData\SSECtest.txt', delimiter='\t', header=None)
df_SSECtrain = pd.read_table('DissertationData\SSECtrain.txt', delimiter='\t', header=None)
df_SemEval2018_EC = pd.read_table('DissertationData\SemEval2018_EC_test.txt', delimiter='\t')

"""
#EMOBANKとSemEval2007の一致率を調べるため
pattern = r"<[^>]*?>"
df_SemEval2007test = df_SemEval2007test['text'].str.replace(pattern, '')
df_SemEval2007train = df_SemEval2007train['text'].str.replace(pattern, '')


#EMOBANKとSemEval2007の一致率を調べるため


a = 0
for texta in df_SemEval2007test:
    for textb in df_emobank['text']:
        if(texta == textb):
            a = a+1


"""



"""
#Original VAD score
df_VAD = pd.DataFrame([
    ['anger', 0.167, 0.865, 0.657], 
    ['fear', 0.073, 0.840, 0.293],
    ['joy', 0.980, 0.824, 0.794],
    ['sadness', 0.052, 0.288, 0.164],
        
    ['anticipation', 0.698, 0.539, 0.711],
    ['disgust', 0.052, 0.775, 0.317],
    ['love', 1.000, 0.519, 0.673],
    ['optimism', 0.949, 0.565, 0.814],
    ['pessimism', 0.083, 0.484, 0.264],
    ['surprise', 0.875, 0.875, 0.562],
    ['trust', 0.888, 0.547, 0.741],
    ])
"""

#In the case "No Emotion", we assign (500, 500, 500)
df_VAD = pd.DataFrame([
    ['anger', 167, 865, 657], 
    ['fear', 73, 840, 293],
    ['joy', 980, 824, 794],
    ['sadness', 52, 288, 164],
    
    ['anticipation', 698, 539, 711],
    ['disgust', 52, 775, 317],
    ['love', 1000, 519, 673],
    ['optimism', 949, 565, 814],
    ['pessimism', 83, 484, 264],
    ['surprise', 875, 875, 562],
    ['trust', 888, 547, 741],
    ])




df_ang['V'] = df_VAD.iat[0,1]
df_ang['A'] = df_VAD.iat[0,2]
df_ang['D'] = df_VAD.iat[0,3]

df_fear['V'] = df_VAD.iat[1,1]
df_fear['A'] = df_VAD.iat[1,2]
df_fear['D'] = df_VAD.iat[1,3]

df_joy['V'] = df_VAD.iat[2,1]
df_joy['A'] = df_VAD.iat[2,2]
df_joy['D'] = df_VAD.iat[2,3]

df_sadness['V'] = df_VAD.iat[3,1]
df_sadness['D'] = df_VAD.iat[3,3]

df_WASSA = pd.concat([df_ang, df_fear, df_joy, df_sadness], ignore_index = True)
df_WASSA.columns = ["id","text","label","intensity","V","A","D"]
df_WASSA = df_WASSA[["text","V","A","D"]]

"""
df_WASSA_V = df_WASSA.loc[:,['text','V']]
df_WASSA_A = df_WASSA.loc[:,['text','A']]
df_WASSA_D = df_WASSA.loc[:,['text','D']]
"""

df_emobank['V'] = round((df_emobank['V']-1) /4 * 1000)
df_emobank['A'] = round((df_emobank['A']-1) /4 * 1000)
df_emobank['D'] = round((df_emobank['D']-1) /4 * 1000)

df_emobank = df_emobank[["text","V","A","D"]]
"""
df_emobank_V = df_emobank[["V","text"]]
df_emobank_A = df_emobank[["A","text"]]
df_emobank_D = df_emobank[["D","text"]]
"""

"""
df_all_V = pd.concat([df_WASSA_V,df_emobank_V], ignore_index = True)
df_all_A = pd.concat([df_WASSA_A,df_emobank_A], ignore_index = True)
df_all_D = pd.concat([df_WASSA_D,df_emobank_D], ignore_index = True)
"""

####################################################################




df_SSEC = pd.concat([df_SSECtest, df_SSECtrain], ignore_index = True)
df_SSEC.columns = ["anger","anticipation","disgust","fear","joy","sadness","surprise","trust","text"]


for emotion in ["anger","anticipation","disgust","fear","joy","sadness","surprise","trust"]:
    df_SSEC[emotion] = df_SSEC[emotion].apply(lambda x : 0 if x == '---' else 1)


df_SSEC['V'] = round((df_SSEC['anger']*df_VAD.iat[0,1] + df_SSEC['fear']*df_VAD.iat[1,1] + df_SSEC['joy']*df_VAD.iat[2,1] + 
    df_SSEC['sadness']*df_VAD.iat[3,1] + df_SSEC['anticipation']*df_VAD.iat[4,1] + df_SSEC['disgust']*df_VAD.iat[5,1] + 
    df_SSEC['surprise']*df_VAD.iat[9,1] + df_SSEC['trust']*df_VAD.iat[10,1]) / 
    (df_SSEC['anger'] + df_SSEC['fear'] +df_SSEC['joy'] + df_SSEC['sadness'] + df_SSEC['anticipation'] + 
     df_SSEC['disgust'] + df_SSEC['surprise'] + df_SSEC['trust']))
    
df_SSEC['A'] = round((df_SSEC['anger']*df_VAD.iat[0,2] + df_SSEC['fear']*df_VAD.iat[1,2] + df_SSEC['joy']*df_VAD.iat[2,2] + 
    df_SSEC['sadness']*df_VAD.iat[3,2] + df_SSEC['anticipation']*df_VAD.iat[4,2] + df_SSEC['disgust']*df_VAD.iat[5,2] + 
    df_SSEC['surprise']*df_VAD.iat[9,2] + df_SSEC['trust']*df_VAD.iat[10,2]) / 
    (df_SSEC['anger'] + df_SSEC['fear'] +df_SSEC['joy'] + df_SSEC['sadness'] + df_SSEC['anticipation'] + 
     df_SSEC['disgust'] + df_SSEC['surprise'] + df_SSEC['trust']))

df_SSEC['D'] = round((df_SSEC['anger']*df_VAD.iat[0,3] + df_SSEC['fear']*df_VAD.iat[1,3] + df_SSEC['joy']*df_VAD.iat[2,3] + 
    df_SSEC['sadness']*df_VAD.iat[3,3] + df_SSEC['anticipation']*df_VAD.iat[4,3] + df_SSEC['disgust']*df_VAD.iat[5,3] + 
    df_SSEC['surprise']*df_VAD.iat[9,3] + df_SSEC['trust']*df_VAD.iat[10,3]) / 
    (df_SSEC['anger'] + df_SSEC['fear'] +df_SSEC['joy'] + df_SSEC['sadness'] + df_SSEC['anticipation'] + 
     df_SSEC['disgust'] + df_SSEC['surprise'] + df_SSEC['trust']))

df_SSEC = df_SSEC[["text","V","A","D"]]










df_SemEval2018_EC = df_SemEval2018_EC.rename(columns={'Tweet': 'text'})


df_SemEval2018_EC['V'] = round((df_SemEval2018_EC['anger']*df_VAD.iat[0,1] + df_SemEval2018_EC['fear']*df_VAD.iat[1,1] + df_SemEval2018_EC['joy']*df_VAD.iat[2,1] + 
    df_SemEval2018_EC['sadness']*df_VAD.iat[3,1] + df_SemEval2018_EC['anticipation']*df_VAD.iat[4,1] + df_SemEval2018_EC['disgust']*df_VAD.iat[5,1] + 
    df_SemEval2018_EC['love']*df_VAD.iat[6,1] + df_SemEval2018_EC['optimism']*df_VAD.iat[7,1] + df_SemEval2018_EC['pessimism']*df_VAD.iat[8,1] + 
    df_SemEval2018_EC['surprise']*df_VAD.iat[9,1] + df_SemEval2018_EC['trust']*df_VAD.iat[10,1]) / 
    (df_SemEval2018_EC['anger'] + df_SemEval2018_EC['fear'] +df_SemEval2018_EC['joy'] + df_SemEval2018_EC['sadness'] + df_SemEval2018_EC['anticipation'] + 
     df_SemEval2018_EC['disgust'] + df_SemEval2018_EC['love'] + df_SemEval2018_EC['optimism'] + df_SemEval2018_EC['pessimism'] + 
     df_SemEval2018_EC['surprise'] + df_SemEval2018_EC['trust']))

df_SemEval2018_EC['A'] = round((df_SemEval2018_EC['anger']*df_VAD.iat[0,2] + df_SemEval2018_EC['fear']*df_VAD.iat[1,2] + df_SemEval2018_EC['joy']*df_VAD.iat[2,2] + 
    df_SemEval2018_EC['sadness']*df_VAD.iat[3,2] + df_SemEval2018_EC['anticipation']*df_VAD.iat[4,2] + df_SemEval2018_EC['disgust']*df_VAD.iat[5,2] + 
    df_SemEval2018_EC['love']*df_VAD.iat[6,2] + df_SemEval2018_EC['optimism']*df_VAD.iat[7,2] + df_SemEval2018_EC['pessimism']*df_VAD.iat[8,2] + 
    df_SemEval2018_EC['surprise']*df_VAD.iat[9,2] + df_SemEval2018_EC['trust']*df_VAD.iat[10,2]) / 
    (df_SemEval2018_EC['anger'] + df_SemEval2018_EC['fear'] +df_SemEval2018_EC['joy'] + df_SemEval2018_EC['sadness'] + df_SemEval2018_EC['anticipation'] + 
     df_SemEval2018_EC['disgust'] + df_SemEval2018_EC['love'] + df_SemEval2018_EC['optimism'] + df_SemEval2018_EC['pessimism'] + 
     df_SemEval2018_EC['surprise'] + df_SemEval2018_EC['trust']))

df_SemEval2018_EC['D'] = round((df_SemEval2018_EC['anger']*df_VAD.iat[0,3] + df_SemEval2018_EC['fear']*df_VAD.iat[1,3] + df_SemEval2018_EC['joy']*df_VAD.iat[2,3] + 
    df_SemEval2018_EC['sadness']*df_VAD.iat[3,3] + df_SemEval2018_EC['anticipation']*df_VAD.iat[4,3] + df_SemEval2018_EC['disgust']*df_VAD.iat[5,3] + 
    df_SemEval2018_EC['love']*df_VAD.iat[6,3] + df_SemEval2018_EC['optimism']*df_VAD.iat[7,3] + df_SemEval2018_EC['pessimism']*df_VAD.iat[8,3] + 
    df_SemEval2018_EC['surprise']*df_VAD.iat[9,3] + df_SemEval2018_EC['trust']*df_VAD.iat[10,3]) / 
    (df_SemEval2018_EC['anger'] + df_SemEval2018_EC['fear'] +df_SemEval2018_EC['joy'] + df_SemEval2018_EC['sadness'] + df_SemEval2018_EC['anticipation'] + 
     df_SemEval2018_EC['disgust'] + df_SemEval2018_EC['love'] + df_SemEval2018_EC['optimism'] + df_SemEval2018_EC['pessimism'] + 
     df_SemEval2018_EC['surprise'] + df_SemEval2018_EC['trust']))


df_SemEval2018_EC = df_SemEval2018_EC[["text","V","A","D"]]





df_train = pd.concat([df_emobank, df_SSEC, df_SemEval2018_EC], ignore_index = True)
df_test = df_WASSA













############################################    BASIC PREPROCESSING     ####################################################################


#Define the pattern which remove from text
mention_re = re.compile(r"@\w+")
url_re = re.compile(r"(?:https?://)?(?:[-\w]+\.)+[a-zA-Z]{2,9}[-\w/#~:;.?+=&%@~]*")




df_train['text'] = df_train['text'].str.replace(mention_re, '')
df_train['text'] = df_train['text'].str.replace(url_re, '')

df_test['text'] = df_test['text'].str.replace(mention_re, '')
df_test['text'] = df_test['text'].str.replace(url_re, '')














############################################    TOKENIZATION AND MORE    #########################################################################
# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")



from transformers import BertTokenizer

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)





"""

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

# For every sentence...
for sent in df_train['text']:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    #To Create the attention masks which explicitly differentiate real tokens from [PAD] tokens, This code uses tokenizer.encode_plus rather than tokenizer.encode.

    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 200,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])


# Convert the lists into tensors.
#add ".to(dev)" when do not use data leader データローダを使わないときは.to(dev)を各行の末尾に付ける
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)


#Change labels depends on each "VAD"
labels = torch.tensor(df_train["V"]).long()
#print(labels)






from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# define data set class 
dataset = TensorDataset(input_ids, attention_masks, labels)

"""

def BERT_tokenization(df):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []
    
    # For every sentence...
    for sent in df['text']:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        #To Create the attention masks which explicitly differentiate real tokens from [PAD] tokens, This code uses tokenizer.encode_plus rather than tokenizer.encode.
    
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 200,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )
        
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
    
    
    # Convert the lists into tensors.
    #add ".to(dev)" when do not use data leader データローダを使わないときは.to(dev)を各行の末尾に付ける
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks



from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

input_ids, attention_masks = BERT_tokenization(df_train)
labels = torch.tensor(df_train["V"]).long()
train_dataset = TensorDataset(input_ids, attention_masks, labels)

input_ids, attention_masks = BERT_tokenization(df_test)
labels = torch.tensor(df_test["V"]).long()
test_dataset = TensorDataset(input_ids, attention_masks, labels)

############################################    TRAIN & TEST    #########################################################################



"""
# get the point of 90% ID 90%地点のIDを取得
train_size = int(0.9* len(dataset))
val_size = len(dataset) - train_size

# divede data set
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
"""


print('Train data：{}'.format(train_size))
print('Validation data:　{} '.format(val_size))

# construct  data loader
#16 is minimum 、64,128 seem to good size
batch_size = 32

# train data loader
train_dataloader = DataLoader(
            train_dataset,  
            sampler = RandomSampler(train_dataset), # random sampling and batch
            batch_size = batch_size
        )

# validation data loader
validation_dataloader = DataLoader(
            val_dataset, 
            sampler = SequentialSampler(val_dataset), # sequential sampling and batch
            batch_size = batch_size
        )



############################################    MODELING   #########################################################################


"""


#Model
#input_ids = torch.tensor(padded)  
input_ids = torch.cat(input_ids, dim=0)
attention_mask = torch.tensor(attention_mask)
"""
from transformers import BertForSequenceClassification, AdamW, BertConfig

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 1000, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model.cuda()




# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    
    


# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )



from transformers import get_linear_schedule_with_warmup

# Number of training epochs. The BERT authors recommend between 2 and 4. 
# We chose to run for 4, but we'll see later that this may be over-fitting the
# training data.
epochs = 4

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)






# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))



import random


# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

# For each epoch...
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode. Don't be mislead--the call to 
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch).
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # It returns different numbers of parameters depending on what arguments
        # arge given and what flags are set. For our useage here, it returns
        # the loss (because we provided labels) and the "logits"--the model
        # outputs prior to activation.
        loss, logits = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using 
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            (loss, logits) = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))







# Display floats with two decimal places.
pd.set_option('precision', 2)

# Create a DataFrame from our training statistics.
df_stats = pd.DataFrame(data=training_stats)

# Use the 'epoch' as the row index.
df_stats = df_stats.set_index('epoch')

# A hack to force the column headers to wrap.
#df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

# Display the table.
df_stats

import matplotlib.pyplot as plt


import seaborn as sns

# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)

# Plot the learning curve.
plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

# Label the plot.
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks([1, 2, 3, 4])

plt.show()



############################################    PREDICTION   #########################################################################


# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

# Put model in evaluation mode
model.eval()

# Tracking variables 
predictions , true_labels , predicted_label = [], [], []

# Predict 
for batch in validation_dataloader:
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)
  
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch
  
  # Telling the model not to compute or store gradients, saving memory and 
  # speeding up prediction
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

  logits = outputs[0]

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
  
  for logit in logits:
      predictions.append(logit)
      predicted_label.append(np.argmax(logit))
      
  for label_id in label_ids:
      true_labels.append(label_id)
  
  """
  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)
  """

print('    DONE.')


predicted_df = pd.DataFrame(predicted_label)
true_df = pd.DataFrame(true_labels)
result_df = pd.concat([predicted_df, true_df], axis=1)

#一致率を計算するとき、一度ndarrayに変換してイコールで結ぶと要素ごとの一致を計算してくれる。
accuracy = np.sum(np.array(predicted_label) == np.array(true_labels))/ len(true_labels)
print('Accuracy : ', accuracy)







############################################    CLASSIFICATION & EVALUATE   #########################################################################


import math

df_VAD = pd.DataFrame([
    ['anger', 167, 865, 657], 
    ['fear', 73, 840, 293],
    ['joy', 980, 824, 794],
    ['sadness', 52, 288, 164],
    ])

    

predicted_labels = []
for line in predicted_df:
    for emotion in df_VAD:
        min_distance = 1000
        distance = math.sqrt((emotion[1] - predicted_df['V']) ** 2 + (emotion[2] - predicted_df['A']) ** 2 + (emotion[3] - predicted_df['D']) ** 2)
        if distance < min_distance:
            min_distance = distance
            predicted_label = emotion[0]
    predicted_labels.append(predicted_label)
            
        
    





























































"""
# Create sentence and label lists
sentences = df.sentence.values
labels = df.label.values

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

# For every sentence...
for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Set the batch size.  
batch_size = 32  

# Create the DataLoader.
prediction_data = TensorDataset(input_ids, attention_masks, labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)



"""





























































"""
model.cuda()
def train(model):
    
    model.train() # train mode
    for batch in train_dataloader:# train_dataloader outputs "word_id", "mask, label"
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        #b_labels = batch[2].to(dev)
        last_hidden_states = model(b_input_ids,  
                             attention_mask=b_input_mask)
        
    return last_hidden_states

last = train(model)
features = last[0][:,0,:].cpu().detach().numpy()


# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    
"""



"""
with torch.no_grad():
    model.cuda()
    last_hidden_states = model(input_ids=input_ids, attention_mask=attention_masks)
"""
    
"""   
features = last_hidden_states[0][:,0,:].numpy()

labels = df_WASSA_V["V"]
"""

#torch.cuda.empty_cache()

#train = df_WASSA.iloc[0:round(len(df_WASSA)*0.8),:]
#test = df_WASSA.iloc[round(len(df_WASSA)*0.8):,:]



















"""
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


import tensorflow as tf
mnist = tf.keras.datasets.mnist
 
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
 
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
 
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

"""
