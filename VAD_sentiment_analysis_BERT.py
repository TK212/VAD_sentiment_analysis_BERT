# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 16:00:47 2018

@author: Ryutaro Takanami
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')


df_ang = pd.read_table('DissertationData\WASSA2017anger.txt', delimiter='\t', header=None)
df_fear = pd.read_table('DissertationData\WASSA2017fear.txt', delimiter='\t', header=None)
df_joy = pd.read_table('DissertationData\WASSA2017joy.txt', delimiter='\t', header=None)
df_sadness = pd.read_table('DissertationData\WASSA2017sadness.txt', delimiter='\t', header=None)




df_VAD = pd.DataFrame([
    ['anger', 0.167, 0.865, 0.657], 
    ['fear', 0.073, 0.840, 0.293],
    ['joy', 0.980, 0.824, 0.794],
    ['sadness', 0.052, 0.288, 0.164],
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
df_sadness['A'] = df_VAD.iat[3,2]
df_sadness['D'] = df_VAD.iat[3,3]

df_WASSA = pd.concat([df_ang, df_fear, df_joy, df_sadness], ignore_index = True)
df_WASSA_V = df_WASSA.loc[:,[1,'V']]
df_WASSA_A = df_WASSA.loc[:,[1,'A']]
df_WASSA_D = df_WASSA.loc[:,[1,'D']]




## Want BERT instead of distilBERT? Uncomment the following line:
model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

"""
#Tokenization
tokenized = df_WASSA_V[1].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))


#Padding
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

#Masking
attention_mask = np.where(padded != 0, 1, 0)
"""


"""
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(dev)
"""

"""
#connect data frames

"""


"""
#PREPROCESSING
DO
remove "mention"
transform "haaaaaaaaappy" to "happy"
remove url



DON'T
don't take up "hashtag" (some paper said it is useful to detect emotion)
do not change to small character
keep emoticon or emoji (that's helpful)


"""




# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

# For every sentence...
for sent in df_WASSA_V[1]:
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
#add ".to(dev)" when do not use data leader データローダを使わないときは.to(dev)を各行の末尾に付ける
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(df_WASSA_V["V"])




from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# define data set class 
dataset = TensorDataset(input_ids, attention_masks, labels)












# get the point of 90% ID 90%地点のIDを取得
train_size = int(0.9* len(dataset))
val_size = len(dataset) - train_size

# divede data set
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

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


"""


#Model
#input_ids = torch.tensor(padded)  
input_ids = torch.cat(input_ids, dim=0)
attention_mask = torch.tensor(attention_mask)
"""

model.cuda()
def train(model):
    
    model.train() # train mode
    for batch in train_dataloader:# train_dataloader outputs "word_id", "mask, label"
        b_input_ids = batch[0].to(dev)
        b_input_mask = batch[1].to(dev)
        #b_labels = batch[2].to(dev)
        last_hidden_states = model(b_input_ids,  
                             attention_mask=b_input_mask)
        
    return last_hidden_states

last = train(model)
features = last[0][:,0,:].cpu().detach().numpy()
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
