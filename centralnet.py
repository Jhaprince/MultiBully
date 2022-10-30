from emotion import Emotion
from sentiment import Sentiment
from sarcasm import Sarcasm
from bully import Bully





class MM(nn.Module):
    def __init__(self, n_out):
        super(MM, self).__init__()  
        
        self.fc_q=nn.Linear(1536,512)
        self.fc_k=nn.Linear(1536,512)
        self.fc_v=nn.Linear(1536,512)
        
        self.bilstm=nn.LSTM(input_size=768,hidden_size=768,bidirectional=True)
        
        
        self.emotion = Emotion(10)
        self.sentiment = Sentiment(3)
        self.bully = Bully(2)
        self.sarcasm = Sarcasm(2)
        
        self.dense_vgg_1024 = nn.Linear(4096, 1024)
        self.dense_vgg_512 = nn.Linear(1024, 512)
        self.drop20 = nn.Dropout(p=0.2)
        self.drop5 = nn.Dropout(p=0.05) 
        
        self.dense_drob_512 = nn.Linear(768, 512)
        
        self.gen_key_L1 = nn.Linear(512, 256) # 512X256
        self.gen_query_L1 = nn.Linear(512, 256) # 512X256
        self.gen_key_L2 = nn.Linear(512, 256) # 512X256
        self.gen_query_L2 = nn.Linear(512, 256) # 512X256
        self.gen_key_L3 = nn.Linear(512, 256) # 512X256
        self.gen_query_L3 = nn.Linear(512, 256) # 512X256
#         self.gen_value = nn.Linear(512, 256) # 512X256
        self.soft = nn.Softmax(dim=1)
        self.soft_final = nn.Softmax(dim=1)
        self.project_dense_512a = nn.Linear(1024, 512) # 512X256
        self.project_dense_512b = nn.Linear(1024, 512) # 512X256
        self.project_dense_512c = nn.Linear(1024, 512) # 512X256 
        
        
        self.fc_out = nn.Linear(512, 256) # 512X256
        self.out = nn.Linear(256, n_out) # 512X256
        
        #attentio fusion prob
        self.prob_img=nn.Linear(1024,1)
        self.prob_text=nn.Linear(1024,1)
        
        #centralnet
        #self.text_512 = nn.Linear(1024,512)
        
        #self.txt_256 = nn.Linear(512,256)
        #self.txt_128 = nn.Linear(256,128)
        #self.txt_out = nn.Linear(128,2)
        
        #self.img_256 = nn.Linear(512,256)
        #self.img_128 = nn.Linear(256,128)
        #self.img_out = nn.Linear(128,2)
        
        
        self.emoti_512 = nn.Linear(1024,512)
        self.emoti_256 = nn.Linear(512,256)
        self.emoti_out = nn.Linear(256,2)
        
        
        self.senti_512 = nn.Linear(1024,512)
        self.senti_256 = nn.Linear(512,256)
        self.senti_out = nn.Linear(256,2)
        
        self.sarcasm_512 = nn.Linear(1024,512)
        self.sarcasm_256 = nn.Linear(512,256)
        self.sarcasm_out = nn.Linear(256,2)
        
        self.img_txt_512 = nn.Linear(1024,512)
        self.img_txt_256 = nn.Linear(512,256)
        self.img_txt_128 = nn.Linear(256,128)
        self.img_txt_out = nn.Linear(128,2)
        
        self.img_txt_multitask = nn.Linear(256*2,256)
        
        self.bully_512 = nn.Linear(1024,512)
        self.bully_256 = nn.Linear(512,256)
        self.bully_out = nn.Linear(256,2)
        
        
        #alphas
        self.alphas_a = nn.ParameterList([torch.nn.Parameter(torch.rand(1)) for i in range(4)])
        self.alphas_v = nn.ParameterList([torch.nn.Parameter(torch.rand(1)) for i in range(4)])
        self.alphas_c = nn.ParameterList([torch.nn.Parameter(torch.rand(1)) for i in range(4)])
        self.alphas_s = nn.ParameterList([torch.nn.Parameter(torch.rand(1)) for i in range(4)])
        self.alphas_t = nn.ParameterList([torch.nn.Parameter(torch.rand(1)) for i in range(4)])

        

    def selfattNFuse_L1a(self, vec1, vec2): 
            q1 = F.relu(self.gen_query_L1(vec1))
            k1 = F.relu(self.gen_key_L1(vec1))
            q2 = F.relu(self.gen_query_L1(vec2))
            k2 = F.relu(self.gen_key_L1(vec2))
            score1 = torch.reshape(torch.bmm(q1.view(-1, 1, 256), k2.view(-1, 256, 1)), (-1, 1))
            score2 = torch.reshape(torch.bmm(q2.view(-1, 1, 256), k1.view(-1, 256, 1)), (-1, 1))
            wt_score1_score2_mat = torch.cat((score1, score2), 1)
            wt_i1_i2 = self.soft(wt_score1_score2_mat.float()) #prob
            prob_1 = wt_i1_i2[:,0]
            prob_2 = wt_i1_i2[:,1]
            wtd_i1 = vec1 * prob_1[:, None]
            wtd_i2 = vec2 * prob_2[:, None]
            
            #print(wtd_i1.shape)
            #print(wtd_i2.shape)
            
            out_rep = F.relu(self.project_dense_512a(torch.cat((wtd_i1,wtd_i2), 1)))
            return out_rep
    def selfattNFuse_L1b(self, vec1, vec2): 
            q1 = F.relu(self.gen_query_L2(vec1))
            k1 = F.relu(self.gen_key_L2(vec1))
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
            out_rep = F.relu(self.project_dense_512b(torch.cat((wtd_i1,wtd_i2), 1)))
            return out_rep
    
    def selfattNFuse_L2(self, vec1, vec2): 
            q1 = F.relu(self.gen_query_L3(vec1))
            k1 = F.relu(self.gen_key_L3(vec1))
            q2 = F.relu(self.gen_query_L3(vec2))
            k2 = F.relu(self.gen_key_L3(vec2))
            score1 = torch.reshape(torch.bmm(q1.view(-1, 1, 256), k2.view(-1, 256, 1)), (-1, 1))
            score2 = torch.reshape(torch.bmm(q2.view(-1, 1, 256), k1.view(-1, 256, 1)), (-1, 1))
            wt_score1_score2_mat = torch.cat((score1, score2), 1)
            wt_i1_i2 = self.soft(wt_score1_score2_mat.float()) #prob
            prob_1 = wt_i1_i2[:,0]
            prob_2 = wt_i1_i2[:,1]
            wtd_i1 = vec1 * prob_1[:, None]
            wtd_i2 = vec2 * prob_2[:, None]
            #out_rep = F.relu(self.project_dense_512c(torch.cat((wtd_i1,wtd_i2), 1)))
            out_rep = torch.cat((wtd_i1,wtd_i2), 1)
            return out_rep
            
    def attention_fusion(self,vec1,vec2):
            img_text=torch.cat((vec1,vec2),1)
            prob_img=F.sigmoid(self.prob_img(img_text))
            prob_txt=F.sigmoid(self.prob_text(img_text))
            
            vec1 = prob_img*vec1
            vec2 = prob_txt*vec2
            
            img_text=torch.cat((vec1,vec2),1)
            
            out_rep=F.relu(self.project_dense_512c(img_text))
            
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
    #def forward(self, in_VGG, in_CI, in_Drob, in_CT):        
        #print(in_CT.shape)
        #VGG_feat = self.drop20(F.relu(self.dense_vgg_512(self.drop20(F.relu(self.dense_vgg_1024(in_VGG))))))
        #in_CI = self.drop20(F.relu(self.dense_vgg_512(self.drop20(F.relu(self.dense_vgg_1024(in_CI))))))
        #in_CT = self.selfattNbistlm(in_CT)
        #VGG_feat=self.drop20(F.relu(self.dense_vgg_512(in_VGG)))
        #Drob_feat = self.drop5(F.relu(self.dense_drob_512(in_Drob)))
        #out_img = self.selfattNFuse_L1a(VGG_feat, in_CI)
        #out_txt = self.selfattNFuse_L1b(Drob_feat, in_CT)        
        #out_img_txt = self.selfattNFuse_L2(out_img, out_txt)
        #out_img_txt = self.selfattNFuse_L2(in_CI,in_CT)
        
        #out_img_txt = self.selfattNFuse_L2(in_CT,in_CI)
        
        #emoti_a = self.emoti_512(out_img_txt)
        #emoti_b = self.emoti_256(emoti_a)
        #emoti_out = self.emoti_out(emoti_b)
        
        #senti_a = self.senti_512(out_img_txt)
        #senti_b = self.senti_256(senti_a)
        #senti_out = self.senti_out(senti_b)
        
        #sar_a = self.sarcasm_512(out_img_txt)
        #sar_b = self.sarcasm_256(sar_a)
        #sar_out =self.sarcasm_out(sar_b)
        
        
        #bully_a = self.bully_512(out_img_txt)
        #bully_b = self.bully_256(bully_a)
        
        #bully_out = self.bully_out(self.img_txt_multitask(torch.cat((emoti_b,bully_b),1)))
        
        
        
        
        #txt = F.relu(self.txt_128(F.relu(self.txt_256(in_CT))))
        #img = F.relu(self.img_128(F.relu(self.img_256(in_CI))))
        
        #self.emotion.load_state_dict(torch.load("path_to_saved_files/Emotion_ModelCkpt/EMNLP_MCHarm_GLAREAll_COVTrain_POLEval/checkpoint_EMNLP_MCHarm_GLAREAll_COVTrain_POLEval.pt"))
        
        #for p in self.emotion.parameters():
        #    p.requires_grad = False
            
            
               
        #self.sarcasm.load_state_dict(torch.load("path_to_saved_files/Sarcasm_ModelCkpt/EMNLP_MCHarm_GLAREAll_COVTrain_POLEval/checkpoint_EMNLP_MCHarm_GLAREAll_COVTrain_POLEval.pt"))
        
        #for p in self.sarcasm.parameters():
        #    p.requires_grad = False
            
            
        #self.sentiment.load_state_dict(torch.load("path_to_saved_files/Sentiment_ModelCkpt/EMNLP_MCHarm_GLAREAll_COVTrain_POLEval/checkpoint_EMNLP_MCHarm_GLAREAll_COVTrain_POLEval.pt"))
        
        #for p in self.sentiment.parameters():
        #    p.requires_grad = False
            
        #self.bully.load_state_dict(torch.load("path_to_saved_files/Bully_ModelCkpt/EMNLP_MCHarm_GLAREAll_COVTrain_POLEval/checkpoint_EMNLP_MCHarm_GLAREAll_COVTrain_POLEval.pt"))
        
        #for p in self.bully.parameters():
        #    p.requires_grad = False
        
        
        
        #concat_emo,emoti_a,emoti_b,emoti_out = self.emotion( in_CI, in_VGG, in_CT, in_Drob)
        #concat_senti,senti_a,senti_b,senti_out = self.sentiment( in_CI, in_VGG, in_CT, in_Drob)
        concat_sar,sar_a,sar_b,sar_out = self.sarcasm( in_CI, in_VGG, in_CT, in_Drob)
        concat_bully,bully_a,bully_b,bully_out = self.bully( in_CI, in_VGG, in_CT, in_Drob)
        
        
        
        sz=in_CI.shape[0]
        
        #emoti_a = self.emoti_512(concat)
        #emoti_b = self.emoti_256(emoti_a)
        #emoti_out = self.emoti_out(emoti_b)
        
        central_rep =torch.zeros((sz,512)).to(device)
        #central_rep =torch.zeros((sz,256)).to(device)
        
        #bully_a = self.bully_512(concat)
        #bully_b = self.bully_256(bully_a)
        #bully_out = self.bully_out(bully_b)
        
        #print(emoti_a.shape)
        
        #xsum_a = concat*self.alphas_a[0].expand_as(concat) + central_rep*self.alphas_c[0].expand_as(central_rep)
        #central_rep = self.img_txt_512(xsum_a)
        
        #xsum_b = emoti_a*self.alphas_a[1].expand_as(emoti_a)+central_rep*self.alphas_c[1].expand_as(central_rep) + bully_a*self.alphas_v[1].expand_as(bully_a)
        #xsum_b = senti_a*self.alphas_a[1].expand_as(senti_a)+central_rep*self.alphas_c[1].expand_as(central_rep) + bully_a*self.alphas_v[1].expand_as(bully_a)
        xsum_b = sar_a*self.alphas_a[1].expand_as(sar_a)+central_rep*self.alphas_c[1].expand_as(central_rep) + bully_a*self.alphas_v[1].expand_as(bully_a)
        #xsum_b = emoti_a*self.alphas_a[1].expand_as(emoti_a)+central_rep*self.alphas_c[1].expand_as(central_rep) + bully_a*self.alphas_v[1].expand_as(bully_a) + senti_a*self.alphas_s[1].expand_as(senti_a)
        #xsum_b = emoti_a*self.alphas_a[1].expand_as(emoti_a)+central_rep*self.alphas_c[1].expand_as(central_rep) + bully_a*self.alphas_v[1].expand_as(bully_a) + sar_a*self.alphas_s[1].expand_as(sar_a)
        #xsum_b = senti_a*self.alphas_a[1].expand_as(senti_a)+central_rep*self.alphas_c[1].expand_as(central_rep) + bully_a*self.alphas_v[1].expand_as(bully_a) + sar_a*self.alphas_s[1].expand_as(sar_a)
        #xsum_b = senti_a*self.alphas_a[1].expand_as(senti_a)+central_rep*self.alphas_c[1].expand_as(central_rep) + bully_a*self.alphas_v[1].expand_as(bully_a) + sar_a*self.alphas_s[1].expand_as(sar_a) + emoti_a*self.alphas_t[1].expand_as(emoti_a)
        
        central_rep = self.img_txt_256(xsum_b)
        #central_rep = self.img_txt_128(xsum_b)
        
        #xsum_c = emoti_b*self.alphas_a[2].expand_as(emoti_b)+central_rep*self.alphas_c[2].expand_as(central_rep) + bully_b*self.alphas_v[2].expand_as(bully_b)
        #xsum_c = senti_b*self.alphas_a[2].expand_as(senti_b)+central_rep*self.alphas_c[2].expand_as(central_rep) + bully_b*self.alphas_v[2].expand_as(bully_b)
        xsum_c = sar_b*self.alphas_a[2].expand_as(sar_b)+central_rep*self.alphas_c[2].expand_as(central_rep) + bully_b*self.alphas_v[2].expand_as(bully_b)
        #xsum_c = emoti_b*self.alphas_a[2].expand_as(emoti_b)+central_rep*self.alphas_c[2].expand_as(central_rep) + bully_b*self.alphas_v[2].expand_as(bully_b) + senti_b*self.alphas_s[2].expand_as(senti_b)
        #xsum_c = emoti_b*self.alphas_a[2].expand_as(emoti_b)+central_rep*self.alphas_c[2].expand_as(central_rep) + bully_b*self.alphas_v[2].expand_as(bully_b) + sar_b*self.alphas_s[2].expand_as(sar_b)
        #xsum_c = senti_b*self.alphas_a[2].expand_as(senti_b)+central_rep*self.alphas_c[2].expand_as(central_rep) + bully_b*self.alphas_v[2].expand_as(bully_b) + sar_b*self.alphas_s[2].expand_as(sar_b)
        #xsum_c = senti_b*self.alphas_a[2].expand_as(senti_b)+central_rep*self.alphas_c[2].expand_as(central_rep) + bully_b*self.alphas_v[2].expand_as(bully_b) + sar_b*self.alphas_s[2].expand_as(sar_b) + emoti_b*self.alphas_t[2].expand_as(emoti_b)
        
        
        
        central_out = self.out(xsum_c)
        #central_out = self.img_txt_out(xsum_c)
        
        
        
        
        
        #out_img_txt = F.relu(self.img_txt_256(F.relu(self.img_txt_512((torch.cat((in_CI,in_CT),1))))))
        #emoti =F.relu(self.emoti_256(F.relu(self.emoti_512((torch.cat((in_CI,in_CT),1))))))
        #sarcasm = F.relu(self.sarcasm_256(F.relu(self.sarcasm_512((torch.cat((in_CI,in_CT),1))))))
        
        #img_txt_multitask = F.relu(self.img_txt_multitask((torch.cat((out_img_txt,emoti,sarcasm),1))))
        
        
   
        #concat=torch.cat((img,txt),1)
        
        
        #out_img_txt = torch.cat((out_img_txt,concat),1)
        #out_img_txt=F.relu(self.dense_vgg_256(out_img_txt))
        #out_img_txt=self.attention_fusion(out_img,out_txt)
        #final_out = F.relu(self.fc_out(out_img_txt))
#         out = torch.sigmoid(self.out(final_out)) #For binary case
        #out = self.out(img_txt_multitask)
        #emoti_out = self.emoti_out(emoti)
        #sarcasm_out =self.sarcasm_out(sarcasm)
        #return F.softmax(self.img_out(img)),F.softmax(self.txt_out(txt)),F.softmax(self.out(final_out))
        #return emoti_out,bully_out
        #return central_out
        #return emoti_out,bully_out,central_out
        #return senti_out,bully_out,central_out
        return sar_out,bully_out,central_out
