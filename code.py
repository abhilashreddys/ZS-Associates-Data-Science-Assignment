import os
import numpy as np
import pandas as pd
import re
import nltk
import inflect 
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from string import punctuation
import string
import tensorflow_hub as hub
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

#Downloading
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')

# Reading Data
df = pd.read_csv('dataset/train_file.csv')
df_test = pd.read_csv('dataset/test_file.csv')

# Cleaning Data
def clean_text(text):
    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    # new_string = [] 
    # p = inflect.engine() 
    # for word in text: 
    #     # if word is a digit, convert the digit 
    #     # to numbers and append into the new_string list 
    #     if word.isdigit(): 
    #         temp = p.number_to_words(word) 
    #         new_string.append(temp) 
  
    #     # append the word as it is 
    #     else: 
    #         new_string.append(word)
    # text = new_string
    ## Remove stop words
    # stops = set(stopwords.words("english"))
    # text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ", text)
    text = re.sub(r"\+", " ", text)
    text = re.sub(r"\-", " ", text)
    text = re.sub(r"\=", " ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r":", " ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(' +', ' ', text) 
    ## Stemming
    # text = text.split()
    # stemmer = SnowballStemmer('english')
    # stemmed_words = [stemmer.stem(word) for word in text]
    # text = " ".join(stemmed_words)

    # lemmatizer=WordNetLemmatizer()
    # text=word_tokenize(text)
    # lemma = [lemmatizer.lemmatize(word) for word in text]
    # text = " ".join(lemma)
    return text

df['Headline']=df['Headline'].apply(clean_text)
df['Title']=df['Title'].apply(clean_text)
df_test['Headline']=df_test['Headline'].apply(clean_text)
df_test['Title']=df_test['Title'].apply(clean_text)

# pandas df to list
title_sent = df['Title'].tolist()
head_sent = df['Headline'].tolist()
senti_title = df['SentimentTitle'].tolist()
senti_head = df['SentimentHeadline'].tolist()
title_sent_test = df_test['Title'].tolist()
head_sent_test = df_test['Headline'].tolist()

# using universal sentence encoder for embedding sentences
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
title_embed = embed(title_sent)
head_embed = embed(head_sent)
title_embed_test = embed(title_sent_test)
head_embed_test = embed(head_sent_test)

# dataset class for dataloader
class tweetdata(Dataset):
    def __init__(self,data_x,data_y):
        self.x = data_x
        self.y = data_y
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]

# using MLP
class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 64)
        self.fc3 = torch.nn.Linear(64,2)
        # self.metrics = 0.0
        # self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(self.relu(out))
        out = self.fc2(out)
        out = self.dropout(self.relu(out))
        out = self.fc3(out)
        # output = self.sigmoid(output)
        return out

# Making data suitable for training and spliting train and val
temp = np.concatenate((np.array(title_embed),np.array(head_embed)),axis=1)
temp_y = np.concatenate((np.array(senti_title).reshape((55932,1)),np.array(senti_head).reshape((55932,1))),axis=1)
# Adding one to sentiment to make neutral around 1
temp_y += np.ones(temp_y.shape)
x_train, x_val, y_train, y_val = train_test_split(temp, temp_y, test_size=0.2, random_state=42)

train_data = tweetdata(x_train,y_train)
val_data = tweetdata(x_val,y_val)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=16,num_workers=0)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=16,num_workers=0)

# Declaring Model and Optimizers
model = MLP(input_size=1024,hidden_size=256)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer) 

# Starting Training
n_epochs = 250
model.train()
for epoch in range(n_epochs):
    train_loss = 0.0
    mae_t = 0.0
    mae_h = 0.0
    for x,y in train_loader:
        x = x.float()
        y = y.float()
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        # loss = criterion(pred[0],y[0])+criterion(pred[1],y[1])
        loss = criterion(pred,y)*10
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        # mae_t += torch.nn.L1Loss(reduction='sum')(pred[0],y[0]).item() 
        # mae_h += torch.nn.L1Loss(reduction='sum')(pred[1],y[1]).item() 
        sc = torch.sum(torch.abs(pred-y),dim=0)
        mae_t += sc[0].item()
        mae_h += sc[-1].item()
        train_loss += loss.item()*x.size(0)
    train_loss = train_loss/len(train_loader.dataset)
    mae_t = mae_t/len(train_loader.dataset)
    mae_h = mae_h/len(train_loader.dataset)
    score = max(0,(1-(0.4*mae_t+0.6*mae_h)))
    print('Epoch: {} \tTraining Loss: {:.6f} \tScore: {:.4f}'.format(epoch+1, train_loss,score))

# Evaluating on Validation
model.eval()
mae_t = 0.0
mae_h = 0.0
for x,y in val_loader:
    x = x.float()
    y = y.float()
    x = x.to(device)
    y = y.to(device)
    pred = model(x) 
    sc = torch.sum(torch.abs(pred-y),dim=0)
    mae_t += sc[0].item()
    mae_h += sc[-1].item()
mae_t = mae_t/len(val_loader.dataset)
mae_h = mae_h/len(val_loader.dataset)
score = max(0,(1-(0.4*mae_t+0.6*mae_h)))
print('Score: {:.4f}'.format(score))

# Saving Model
torch.save(model.state_dict(), 'L10E250.pth')

# Data class for testing
class tweetdatatest(Dataset):
    def __init__(self,data_id,data_x):
        self.id = data_id
        self.x = data_x
        # self.y = data_y
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return self.id[idx],self.x[idx]

# Making data suitable for testing and loading
temp_test = np.concatenate((np.array(title_embed_test),np.array(head_embed_test)),axis=1)
test_ids = df_test['IDLink'].tolist()
test_data = tweetdatatest(test_ids,temp_test)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=4,num_workers=0)


# Testing
modeltest = MLP(input_size=1024,hidden_size=256)
modeltest.load_state_dict(torch.load('L10E250.pth'))

modeltest.eval()
id_list = []
title_list = []
head_list = []
for id,x in test_loader:
    # print(id)
    x = x.float()
    # x = x.to(device)
    pred = modeltest(x)
    id_list += list(id)
    pred.cpu().detach().numpy()
    title_list += pred[:,0].tolist()
    head_list += pred[:,-1].tolist()

# subtracting one to sentiment to make neutral around 0 like the original
og_title = [t-1 for t in title_list]
og_head = [t-1 for t in head_list]

# Saving for submission
test = pd.DataFrame({'IDLink':id_list, 'SentimentTitle':og_title, 'SentimentHeadline':og_head}) 
test.to_csv('submission1.csv',index=False)