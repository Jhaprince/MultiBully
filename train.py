import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
#from torchnlp import encoders
from PIL import Image
#import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import  mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from pathlib import Path

import os
import io

from matplotlib import pyplot as plt
from matplotlib import patches as pch
import pickle
import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re


import torch
from torch import optim, nn
from torchvision import models, transforms
import cv2


from tqdm import tqdm
import numpy as np



from emotion import Emotion
from sarcasm import Sarcasm
from sentiment import Sentiment
from bully import Bully
from centralnet import MM



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



torch.cuda.empty_cache()


#importing the dataset

out_train=open("/DATA/prince_1901cs04/prince/(31 July) IMG-Data/NAACL/Visualbert/dataset/70_10_20/train_id.pkl","rb")
train_id=pickle.load(out_train)

train_id=train_id[:]

out_val=open("/DATA/prince_1901cs04/prince/(31 July) IMG-Data/NAACL/Visualbert/dataset/70_10_20/val_id.pkl","rb")
val_id=pickle.load(out_val)


out_test=open("/DATA/prince_1901cs04/prince/(31 July) IMG-Data/NAACL/Visualbert/dataset/70_10_20/test_id.pkl","rb")
test_id=pickle.load(out_test)




out_proposal=open("/DATA/prince_1901cs04/prince/(31 July) IMG-Data/NAACL/Visualbert/img_feature/vgg19.pkl","rb")  # Masked RCNN Resnet+FPN
proposal=pickle.load(out_proposal)

xlm_roberta=open("/DATA/prince_1901cs04/prince/(31 July) IMG-Data/NAACL/Visualbert/text_feature/mbert.pkl","rb")
xlm_roberta=pickle.load(xlm_roberta)

text_clip_feat=open("/DATA/prince_1901cs04/prince/(31 July) IMG-Data/NAACL/Visualbert/clip/text_clip.pkl","rb")
text_clip_feat=pickle.load(text_clip_feat)


image_clip_feat=open("/DATA/prince_1901cs04/prince/(31 July) IMG-Data/NAACL/Visualbert/clip/image_clip.pkl","rb")
image_clip_feat=pickle.load(image_clip_feat)

data=open("/DATA/prince_1901cs04/prince/(31 July) IMG-Data/NAACL/Visualbert/dataset/data-corrected.pkl","rb")
data=pickle.load(data)



temp=open("/DATA/prince_1901cs04/prince/(31 July) IMG-Data/NAACL/Visualbert/dataset/data.pkl","rb")
temp=pickle.load(temp)


data['2325.jpg']=temp['2325.jpg']









"""
 MODELS = {
     "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
     "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
     "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
     "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"   
 }
! wget {MODELS["ViT-B/32"]} -O clip_model.pt
"""



clip_model = torch.jit.load("/DATA/prince_1901cs04/prince/(31 July) IMG-Data/NAACL/MOMENTA/pretrained_model/ViT-B-32.pt").cuda().eval()
input_resolution = clip_model.input_resolution.item()
context_length = clip_model.context_length.item()
vocab_size = clip_model.vocab_size.item()

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)



 
class MemesDataset(torch.utils.data.Dataset):
    """Uses jsonl data to preprocess and serve 
    dictionary of multimodal tensors for model input.
    """

    def __init__(
        self,
        data_instance,
        data,
        proposal,
        xlm_roberta,
        text_clip_feat,
        image_clip_feat,
        split_flag=None,
        balance=False,
        dev_limit=None,
        random_state=0,
    ):

        self.samples_frame = data_instance
        self.data=data        
        self.proposal=proposal
        self.xlm_roberta=xlm_roberta
        self.text_clip_feat=text_clip_feat
        self.image_clip_feat=image_clip_feat
        
       
        
    def __len__(self):
        """This method is called when you do len(instance) 
        for an instance of this class.
        """
        return len(self.samples_frame)

    def __getitem__(self, idx):
        """This method is called when you do instance[key] 
        for an instance of this class.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_id=self.samples_frame[idx]
         
        if(self.xlm_roberta[img_id].shape[1]<50):
            sz=50-self.xlm_roberta[img_id].shape[1]
            pad=torch.zeros((1,sz,768))
            self.xlm_roberta[img_id]=torch.cat((self.xlm_roberta[img_id],pad),axis=1)
            
        else:
            self.xlm_roberta[img_id]=self.xlm_roberta[img_id][:,:50,:]
            
            
        
        text_drob_feature=torch.tensor(self.xlm_roberta[img_id]).squeeze(0).to(device) #(batch_size,768)
        image_clip_input=torch.tensor(self.image_clip_feat[img_id]).to(device)   #(batch_size,512)
        image_vgg_feature=torch.mean(torch.tensor(self.proposal[img_id]),axis=0).to(device)   # (batch_size,1024)       
        text_clip_input=torch.tensor(self.text_clip_feat[img_id]).to(device)                   #(batch_size,512)
        #text_drob_feature=torch.mean(torch.tensor(self.xlm_roberta[img_id]),axis=1).squeeze(0).to(device) #(batch_size,768)
        label=torch.tensor(self.data[img_id]["label"]).to(device)
        sentiment=torch.tensor(self.data[img_id]["sentiment"]).to(device)
        sarcasm=torch.tensor(self.data[img_id]["sarcasm"]).to(device)
        emotion=torch.tensor(self.data[img_id]["emotion"]).to(device)
        harm_score=torch.tensor(self.data[img_id]["harmful-score"]).to(device)
        target=torch.tensor(self.data[img_id]["target"]).to(device)
        
       
       
        sample = {
            "id": img_id, 
            "image_clip_input": image_clip_input,
            "image_vgg_feature": image_vgg_feature,
            "text_clip_input": text_clip_input,
            "text_drob_embedding": text_drob_feature,
            "label": label,
            "sentiment":sentiment,
            "sarcasm":sarcasm,
            "emotion":emotion,
            "harmful-score":harm_score,
            "target":target
        }


        return sample





bm_dataset_train = MemesDataset(train_id,data,proposal,xlm_roberta,text_clip_feat,image_clip_feat,split_flag='train')
dataloader_train = DataLoader(bm_dataset_train, batch_size=32,shuffle=False, num_workers=0)

print("train_data loaded")

bm_dataset_val = MemesDataset(val_id,data,proposal,xlm_roberta,text_clip_feat,image_clip_feat,split_flag='val')
dataloader_val = DataLoader(bm_dataset_val, batch_size=32,shuffle=False, num_workers=0)
print("validation_data loaded")


bm_dataset_test = MemesDataset(test_id,data,proposal,xlm_roberta,text_clip_feat,image_clip_feat,split_flag='test')
dataloader_test = DataLoader(bm_dataset_test, batch_size=32,shuffle=False, num_workers=0)
print("test data loaded")




output_size = 2

exp_name = "EMNLP_MCHarm_GLAREAll_COVTrain_POLEval"
# pre_trn_ckp = "EMNLP_MCHarm_GLAREAll_COVTrain" # Uncomment for using pre-trained
exp_path = "path_to_saved_files/EMNLP_ModelCkpt/"+exp_name
#exp_path = "path_to_saved_files/Sarcasm_ModelCkpt/"+exp_name
lr=0.0001
# criterion = nn.BCELoss() #Binary case
criterion = nn.CrossEntropyLoss()

# # ------------Fresh training------------
model = MM(output_size)
#model = Sarcasm(output_size)
model.to(device)

print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# In[ ]:


# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total trainable parameters are: {total_params}")


# ## Training

# https://github.com/Bjarten/early-stopping-pytorch

# In[ ]:


# import EarlyStopping
import os, sys
#sys.path.append('path_to_the_module/early-stopping-pytorch')
from pytorchtools import EarlyStopping
#from torchsample.callbacks import EarlyStopping


# For cross entropy loss
def train_model(model, patience, n_epochs):
    epochs = n_epochs
#     clip = 5

    train_acc_list=[]
    val_acc_list=[]
    train_loss_list=[]
    val_loss_list=[]
    
        # initialize the experiment path
    Path(exp_path).mkdir(parents=True, exist_ok=True)
    # initialize early_stopping object
    chk_file = os.path.join(exp_path, 'checkpoint_'+exp_name+'.pt')
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=chk_file)


    model.train()
    for i in range(epochs):
#         total_acc_train = 0

      
        #scheduler.step()

        total_loss_train = 0
        total_train = 0
        correct_train = 0

        for data in dataloader_train:
            
#             Clip features...
            img_inp_clip = data['image_clip_input']
            txt_inp_clip = data['text_clip_input']
            
            print("clip image",img_inp_clip.shape)
            print("clip text",txt_inp_clip.shape)
            
            with torch.no_grad():
                img_feat_clip = clip_model.encode_image(img_inp_clip).float().to(device)
                txt_feat_clip = clip_model.encode_text(txt_inp_clip).float().to(device)

            img_feat_vgg = data['image_vgg_feature']
            txt_feat_trans = data['text_drob_embedding']

            label_train = data['label'].to(device)
            emotion_train = data['emotion'].to(device)
            sentiment_train = data['sentiment'].to(device)
            sarcasm_train = data['sarcasm'].to(device)

            model.zero_grad()
            #img, txt, output = model(img_feat_clip, img_feat_vgg, txt_feat_clip, txt_feat_trans)
            #concat,emoti_a,emoti_b,output = model(img_feat_clip, img_feat_vgg, txt_feat_clip, txt_feat_trans)
            #output = model(img_feat_clip, img_feat_vgg, txt_feat_clip, txt_feat_trans)
            #emoti_out,output = model(img_feat_clip, img_feat_vgg, txt_feat_clip, txt_feat_trans)
            #emoti_out,bully_out,output = model(img_feat_clip, img_feat_vgg, txt_feat_clip, txt_feat_trans)
            #senti_out,bully_out,output = model(img_feat_clip, img_feat_vgg, txt_feat_clip, txt_feat_trans)
            sar_out,bully_out,output = model(img_feat_clip, img_feat_vgg, txt_feat_clip, txt_feat_trans)
#             print(output.shape)
            #output = model(img_feat_vgg, txt_feat_trans)

            #loss = criterion(output.squeeze(), label_train)+(0.5*criterion(emoti_out.squeeze(), emotion_train))+(0.5*criterion(bully_out.squeeze(), label_train))
            #loss = criterion(output.squeeze(), label_train)+(0.5*criterion(senti_out.squeeze(), sentiment_train))+(0.5*criterion(bully_out.squeeze(), label_train))
            loss = criterion(output.squeeze(), label_train)+(0.5*criterion(sar_out.squeeze(), sarcasm_train))+(0.5*criterion(bully_out.squeeze(), label_train))
            #loss = criterion(output.squeeze(), label_train)+criterion(emoti_out.squeeze(), emotion_train)+criterion(sarcasm_out.squeeze(), sarcasm_train)
            #loss = criterion(output.squeeze(), label_train)+criterion(emoti_out.squeeze(), emotion_train)+criterion(sarcasm_out.squeeze(), sarcasm_train)
            #loss = criterion(output.squeeze(), label_train)+criterion(emoti_out.squeeze(), emotion_train)+criterion(sarcasm_out.squeeze(), sarcasm_train)
            #loss = criterion(output.squeeze(), label_train)+(0.5*criterion(emoti_out.squeeze(), emotion_train))+(0.5*criterion(senti_out.squeeze(), sentiment_train))
            #loss = criterion(output.squeeze(), label_train)+(0.5*criterion(emoti_out.squeeze(), emotion_train))+(0.5*criterion(senti_out.squeeze(), sentiment_train))+(0.5*criterion(senti_out.squeeze(), sentiment_train))
            #loss = criterion(output.squeeze(), label_train)+(0.5*criterion(emoti_out.squeeze(), emotion_train)) 
            #loss = criterion(output.squeeze(), label_train)
            #loss = criterion(output.squeeze(), emotion_train)
            
#             print(loss)
            loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            with torch.no_grad():
                _, predicted_train = torch.max(output.data, 1)
                total_train += label_train.size(0)
                #total_train += emotion_train.size(0)
                correct_train += (predicted_train == label_train).sum().item()
                #correct_train += (predicted_train == emotion_train).sum().item()
#                 out_val = (output.squeeze()>0.5).float()
#                 out_final = ((out_val == 1).nonzero(as_tuple=True)[0])
#                 print()
#                 acc = torch.abs(output.squeeze() - label.float()).view(-1)
#                 acc = (1. - acc.sum() / acc.size()[0])
#                 total_acc_train += acc
                total_loss_train += loss.item()

        
        train_acc = 100 * correct_train / total_train
        train_loss = total_loss_train/total_train
        model.eval()
#         total_acc_val = 0
        total_loss_val = 0
        total_val = 0
        correct_val = 0

        with torch.no_grad():
            for data in dataloader_val:                
#                 Clip features...                
                img_inp_clip = data['image_clip_input']
                txt_inp_clip = data['text_clip_input']
                with torch.no_grad():
                    img_feat_clip = clip_model.encode_image(img_inp_clip).float().to(device)
                    txt_feat_clip = clip_model.encode_text(txt_inp_clip).float().to(device)
                
                
                img_feat_vgg = data['image_vgg_feature']                
                txt_feat_trans = data['text_drob_embedding']

                

                label_val = data['label'].to(device)
                emotion_val = data['emotion'].to(device)
                sentiment_val = data['sentiment'].to(device)
                sarcasm_val = data['sarcasm'].to(device)


                model.zero_grad()
                
                #emoti_out,bully_out,output = model(img_feat_clip, img_feat_vgg, txt_feat_clip, txt_feat_trans)
                #senti_out,bully_out,output = model(img_feat_clip, img_feat_vgg, txt_feat_clip, txt_feat_trans)
                sar_out,bully_out,output = model(img_feat_clip, img_feat_vgg, txt_feat_clip, txt_feat_trans)
                #sar_out,bully_out,output = model(img_feat_clip, img_feat_vgg, txt_feat_clip, txt_feat_trans)
                #img, txt, output = model(img_feat_clip, img_feat_vgg, txt_feat_clip, txt_feat_trans)
                #emoti_out,output = model(img_feat_clip, img_feat_vgg, txt_feat_clip, txt_feat_trans)
                #concat,emoti_a,emoti_b,output = model(img_feat_clip, img_feat_vgg, txt_feat_clip, txt_feat_trans)
                #output = model(img_feat_clip, img_feat_vgg, txt_feat_clip, txt_feat_trans)
#                 output = model(img_feat_vgg, txt_feat_trans)

                #val_loss = criterion(output.squeeze(), label_val)+(0.5*criterion(emoti_out.squeeze(), emotion_val))+(0.5*criterion(bully_out.squeeze(), label_val))
                #val_loss = criterion(output.squeeze(), label_val)+(0.5*criterion(senti_out.squeeze(), sentiment_val))+(0.5*criterion(bully_out.squeeze(), label_val))
                val_loss = criterion(output.squeeze(), label_val)+(0.5*criterion(sar_out.squeeze(), sarcasm_val))+(0.5*criterion(bully_out.squeeze(), label_val))
                
                
                #val_loss = criterion(output.squeeze(), label_val)+criterion(emoti_out.squeeze(), emotion_val)+criterion(sarcasm_out.squeeze(), sarcasm_val)
                #val_loss = criterion(output.squeeze(), label_val)+(0.5*criterion(emoti_out.squeeze(), emotion_val))
                #val_loss = criterion(output.squeeze(), label_val)+(0.5*criterion(emoti_out.squeeze(), emotion_val)) + (0.5*criterion(emoti_out.squeeze(), emotion_val))
                #val_loss = criterion(output.squeeze(), label_val)+(0.5*criterion(emoti_out.squeeze(), emotion_val)) + (0.5*criterion(senti_out.squeeze(), sentiment_val))
                #val_loss = criterion(output.squeeze(), label_val)+(0.5*criterion(emoti_out.squeeze(), emotion_val)) + (0.5*criterion(senti_out.squeeze(), sentiment_val)) + (0.5*criterion(sar_out.squeeze(), sarcasm_val))
                #val_loss = criterion(output.squeeze(), label_val)
                #val_loss = criterion(output.squeeze(), emotion_val)
                
                _, predicted_val = torch.max(output.data, 1)
                total_val += label_val.size(0)
                #total_val += emotion_val.size(0)
                correct_val += (predicted_val == label_val).sum().item()                
                #correct_val += (predicted_val == emotion_val).sum().item()                
                total_loss_val += val_loss.item()


        val_acc = 100 * correct_val / total_val
        val_loss = total_loss_val/total_val

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        
        early_stopping(val_loss, model)
        
        #print("Saving model...") 
        #torch.save(model.state_dict(), os.path.join(exp_path, "final.pt"))
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
            

        
        print(f'Epoch {i+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
        model.train()
        torch.cuda.empty_cache()
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(chk_file))
    #model.load_state_dict(torch.load(os.path.join(exp_path,"final.pt")))
    
    return  model, train_acc_list, val_acc_list, train_loss_list, val_loss_list, i
        


# For CE loss
def test_model(model):
    model.eval()
    total_test = 0
    correct_test =0
    total_acc_test = 0
    total_loss_test = 0
    outputs = []
    test_labels=[]
    with torch.no_grad():
        for data in dataloader_test:
            img_inp_clip = data['image_clip_input']
            txt_inp_clip = data['text_clip_input']
            with torch.no_grad():
                img_feat_clip = clip_model.encode_image(img_inp_clip).float().to(device)
                txt_feat_clip = clip_model.encode_text(txt_inp_clip).float().to(device)

            img_feat_vgg = data['image_vgg_feature']
            txt_feat_trans = data['text_drob_embedding']            

            label_test = data['label'].to(device)
            
#             out = model(img_feat_vgg, txt_feat_trans)        

            #_,out = model(img_feat_clip, img_feat_vgg, txt_feat_clip, txt_feat_trans)        
            _,_,out = model(img_feat_clip, img_feat_vgg, txt_feat_clip, txt_feat_trans)        
            #out = model(img_feat_clip, img_feat_vgg, txt_feat_clip, txt_feat_trans)        

            outputs += list(out.cpu().data.numpy())
            loss = criterion(out.squeeze(), label_test)
            
            _, predicted_test = torch.max(out.data, 1)
            total_test += label_test.size(0)
            correct_test += (predicted_test == label_test).sum().item()
#                 out_val = (output.squeeze()>0.5).float()
#                 out_final = ((out_val == 1).nonzero(as_tuple=True)[0])
#                 print()
#                 acc = torch.abs(output.squeeze() - label.float()).view(-1)
#                 acc = (1. - acc.sum() / acc.size()[0])
#                 total_acc_train += acc
            total_loss_test += loss.item()
            
            
#     #         print(label.float())
#             acc = torch.abs(out.squeeze() - label.float()).view(-1)
#     #         print((acc.sum() / acc.size()[0]))
#             acc = (1. - acc.sum() / acc.size()[0])
#     #         print(acc)
#             total_acc_test += acc
#             total_loss_test += loss.item()

    
    acc_test = 100 * correct_test / total_test
    loss_test = total_loss_test/total_test   
    
    print(f'acc: {acc_test:.4f} loss: {loss_test:.4f}')
    return outputs




n_epochs = 200
# early stopping patience; how long to wait after last time validation loss improved.
patience = 10
model, train_acc_list, val_acc_list, train_loss_list, val_loss_list, epoc_num = train_model(model, patience, n_epochs)
#chk_file = os.path.join(exp_path, 'checkpoint_'+exp_name+'.pt')
#model.load_state_dict(torch.load(chk_file))
# Plot the training and validation curves


# Multiclass setting - Harmful
y_pred=[]
for i in outputs:
#     print(np.argmax(i))
    y_pred.append(np.argmax(i))
# # np.argmax(outputs[:])
# outputs

# # Multiclass setting
test_labels=[]

for index in range(len(test_id)):
    img_id=test_id[index]
    test_labels.append(data[img_id]["label"])
    
 

# In[ ]:


def calculate_mmae(expected, predicted, classes):
    NUM_CLASSES = len(classes)
    count_dict = {}
    dist_dict = {}
    for i in range(NUM_CLASSES):
        count_dict[i] = 0
        dist_dict[i] = 0.0
    for i in range(len(expected)):
        dist_dict[expected[i]] += abs(expected[i] - predicted[i])
        count_dict[expected[i]] += 1
    overall = 0.0
    for claz in range(NUM_CLASSES): 
        class_dist =  1.0 * dist_dict[claz] / count_dict[claz] 
        overall += class_dist
    overall /= NUM_CLASSES
#     return overall[0]
    return overall


# In[ ]:


rec = np.round(recall_score(test_labels, y_pred, average="weighted"),4)
prec = np.round(precision_score(test_labels, y_pred, average="weighted"),4)
f1 = np.round(f1_score(test_labels, y_pred, average="weighted"),4)
# hl = np.round(hamming_loss(test_labels, y_pred),4)
acc = np.round(accuracy_score(test_labels, y_pred),4)
#mmae = np.round(calculate_mmae(test_labels, y_pred, [0,1]),4)
#mae = np.round(mean_absolute_error(test_labels, y_pred),4)
# print("recall_score\t: ",rec)
# print("precision_score\t: ",prec)
# print("f1_score\t: ",f1)
# print("hamming_loss\t: ",hl)
# print("accuracy_score\t: ",f1)

print(classification_report(test_labels, y_pred))



print("Acc, F1, Rec, Prec, MAE, MMAE")
#print(acc, f1, rec, prec, mae, mmae)
print(acc, f1, rec, prec)

