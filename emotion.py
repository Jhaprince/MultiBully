

#This is used to train the model only over the emotion




class Emotion(nn.Module):
    def __init__(self,n_out):
        super(Emotion,self).__init__()
        
        
        self.fc_q=nn.Linear(1536,512)
        self.fc_k=nn.Linear(1536,512)
        self.fc_v=nn.Linear(1536,512)
        
        self.bilstm=nn.LSTM(input_size=768,hidden_size=768,bidirectional=True)
        
        self.dense_vgg_1024 = nn.Linear(4096, 1024)
        self.dense_vgg_512 = nn.Linear(1024, 512)
        self.drop20 = nn.Dropout(p=0.2)
        self.drop5 = nn.Dropout(p=0.05) 
        
        
               
        self.gen_key_L1 = nn.Linear(512, 256) # 512X256
        self.gen_query_L1 = nn.Linear(512, 256) # 512X256
        
        self.gen_key_L2 = nn.Linear(512, 256) # 512X256
        self.gen_query_L2 = nn.Linear(512, 256) # 512X256
        
        self.soft = nn.Softmax(dim=1)
        
        
        self.emoti_512 = nn.Linear(1024,512)
        self.emoti_256 = nn.Linear(512,256)
        self.emoti_128 = nn.Linear(256,128)
        self.emoti_out = nn.Linear(256,n_out)
        self.out = nn.Linear(128,n_out)
        
    def selfattNFuse_L2(self, vec1, vec2): 
        q1 = F.relu(self.gen_query_L1(vec1))
        k1 = F.relu(self.gen_key_L1(vec1))
        q2 = F.relu(self.gen_query_L2(vec2))
        k2 = F.relu(self.gen_key_L2(vec2))
        score1 = torch.reshape(torch.bmm(q1.view(-1, 1, 256), k2.view(-1, 256, 1)), (-1, 1))
        score2 = torch.reshape(torch.bmm(q2.view(-1, 1, 256), k1.view(-1, 256, 1)), (-1, 1))
        wt_score1_score2_mat = torch.cat((score1, score2), 1)
        wt_i1_i2 = self.soft(wt_score1_score2_mat.float()) #prob
        prob_1 = wt_i1_i2[:,0]
        prob_2 = wt_i1_i2[:,1]
        wtd_i1 = vec1 * prob_1[:, None]
        wtd_i2 = vec2 * prob_2[:, None]
        out_rep = torch.cat((wtd_i1,wtd_i2), 1)
  
        return out_rep
    
    def attention_fusion(self,vec1,vec2):
        img_text=torch.cat((vec1,vec2),1)
        prob_img=F.sigmoid(self.prob_img(img_text))
        prob_txt=F.sigmoid(self.prob_text(img_text))
        
        vec1 = prob_img*vec1
        vec2 = prob_txt*vec2
        
        out_rep=torch.cat((vec1,vec2),1)
        
        return out_rep   
        
    def selfattNbistlm(self,input):

        #print(input.shape)
        batch_size = input.shape[0]

        input = input.permute(1,0,2)
        
        
        h0 = c0 = torch.zeros((2,batch_size,768)).to(device)
                
        after_lstm=self.bilstm(input,(h0,c0))[0]

        q = F.relu(self.fc_q(after_lstm).permute(1,0,2))

        k = F.relu(self.fc_k(after_lstm).permute(1,2,0))

        v = F.relu(self.fc_v(after_lstm).permute(1,0,2))

        att = F.tanh(torch.bmm(q,k))

        #print(att_1.shape)

        soft = F.softmax(att,2)

        #print(soft_1.shape)

        value = torch.mean(torch.bmm(soft,v),1)
        
        #print(value.shape)

        return value
                        
    
        
    def forward(self, in_CI, in_VGG, in_CT, in_Drob):
    
        #remove comments from following lines if you want to use BERT + VGG19 combination
   
        #in_CI = self.drop20(F.relu(self.dense_vgg_512(self.drop20(F.relu(self.dense_vgg_1024(in_VGG))))))
        #in_CT = self.selfattNbistlm(in_Drob)    
    
        
   #remove below comments accordingly as per the type of attention you want to give.
        
        #concat = self.selfattNFuse_L2(in_CT,in_CI)
        #concat = self.attention_fusion(in_CT,in_CI)
        concat = torch.cat((in_CT,in_CI),1)
        #concat = in_CT
        emoti_a = self.emoti_512(concat)
        emoti_b = self.emoti_256(emoti_a)
        emoti_out = self.emoti_out(emoti_b)
        
        #emoti_a = self.emoti_256(concat)
        #emoti_b = self.emoti_128(emoti_a)
        #emoti_out = self.out(emoti_b)
        
        return concat,emoti_a,emoti_b,emoti_out
        
