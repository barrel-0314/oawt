from transformers import BertModel,BertTokenizer
import torch
import pickle
import numpy as np
import time
import os
from utils import mention_pre
import csv
from unlinkable_sample import read_new_tableset
import json

def numpy_to_json(obj):
    
    if isinstance(obj,np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def mention_can_diff(men,can,model,tokenizer):
    
    batch=tokenizer([men],padding=True,return_tensors="pt")
    fm = model(batch['input_ids'], batch['attention_mask'], output_hidden_states=True)[1]
    batch=tokenizer([can],padding=True,return_tensors="pt")
    fc = model(batch['input_ids'], batch['attention_mask'], output_hidden_states=True)[1]
    dis=torch.sub(fc,fm)
    N=torch.norm(dis)
    return float(N)
    
def mention_bert(WT,model,tokenizer):
    
    B={}
    for r in range(len(WT['con'])):
        
        for c in range(WT['col_num']):
            
            mention=WT['con'][r][c]
            mention=mention.strip(' ')
            M=mention_pre(mention)
            
            if len(M)>1:
                men=' '.join(M)
                batch=tokenizer([men],padding=True,return_tensors="pt")
                fm = model(batch['input_ids'], batch['attention_mask'], output_hidden_states=True)[1]
                B[(r,c)]=fm
            else:
                batch=tokenizer([M[0]],padding=True,return_tensors="pt")
                fm = model(batch['input_ids'], batch['attention_mask'], output_hidden_states=True)[1]
                B[(r,c)]=fm
    return B

def candidate_bert_score(WT,model,tokenizer,Can,B):
    
    CB={}
    CB['entity']={}
    CB['literal']={}
    for r in range(len(WT['con'])):
        
        for c in range(WT['col_num']):
            mention=WT['con'][r][c]
            mention=mention.strip(' ')
            M=mention_pre(mention)
            if len(M)>1:
                for men in M:
                    
                    if men in Can['entity']:
                        CB['entity'][men]=[]
                        for can in Can['entity'][men]:
                            batch=tokenizer([can],padding=True,return_tensors="pt")
                            fm = model(batch['input_ids'], batch['attention_mask'], output_hidden_states=True)[1]
                            dis=torch.sub(B[(r,c)],fm)
                            N=torch.norm(dis)
                            CB['entity'][men].append(float(N))
                    if men in Can['literal']:
                        CB['literal'][men]=[]
                        for can in Can['literal'][men]:
                            batch=tokenizer([can],padding=True,return_tensors="pt")
                            fm = model(batch['input_ids'], batch['attention_mask'], output_hidden_states=True)[1]
                            dis=torch.sub(B[(r,c)],fm)
                            N=torch.norm(dis)
                            CB['literal'][men].append(float(N))
            else:
                if M[0] in Can['entity']:
                    CB['entity'][M[0]]=[]
                    for can in Can['entity'][M[0]]:
                        batch=tokenizer([can],padding=True,return_tensors="pt")
                        fm = model(batch['input_ids'], batch['attention_mask'], output_hidden_states=True)[1]
                        dis=torch.sub(B[(r,c)],fm)
                        N=torch.norm(dis)
                        CB['entity'][M[0]].append(float(N))
                if M[0] in Can['literal']:
                    CB['literal'][M[0]]=[]
                    for can in Can['literal'][M[0]]:
                        batch=tokenizer([can],padding=True,return_tensors="pt")
                        fm = model(batch['input_ids'], batch['attention_mask'], output_hidden_states=True)[1]
                        dis=torch.sub(B[(r,c)],fm)
                        N=torch.norm(dis)
                        CB['literal'][M[0]].append(float(N))
                    
    return CB
    

def store_candidate_bert_score(TS_name):
    
    model=BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    CBS={}
    with open('Data/'+TS_name+'_table.json','r',encoding='utf-8') as file:
        TableSet=json.load(file)
    with open('Candidate/'+TS_name+'_table_cpl.json','r',encoding='utf-8') as file:
        C=json.load(file)
    #j=0
    for i in TableSet:
        print(i)
        '''
        j+=1
        if j>3:
            break
        '''
        if 1>0:
            W=TableSet[i]
            Can=C[i]
            B=mention_bert(W,model,tokenizer)
            CBS[i]=candidate_bert_score(W,model,tokenizer,Can,B)
            
        else:
            CBS[i]='Fail'
            print('fail')
            
    with open('BERT/'+TS_name+'_cbs2.json','w') as f:
        json.dump(CBS,f)




def bert_score_column_type(WT,model,tokenizer,ctf_can):
    
    ctf_bert={}
    for col in ctf_can:
        
        ctf_bert[col]=[]
        for tup in ctf_can[col]:
            
            Col=[c[int(col)] for c in WT['con'] if int(col)<len(c) and len(c[int(col)])<512]
            #print(Col)
            batch=tokenizer(Col,padding=True,return_tensors="pt")
            fm = model(batch['input_ids'], batch['attention_mask'], output_hidden_states=True)[1]
            
            batch=tokenizer(tup[0],padding=True,return_tensors="pt")
            t_fm = model(batch['input_ids'], batch['attention_mask'], output_hidden_states=True)[1]
            #print(fm.size(),t_fm.size())
            diff_sum=float(torch.mean(torch.sum(torch.pow(fm-t_fm,2),dim=1)))
            ctf_bert[col].append((tup[0],diff_sum))
            #t_v=torch.tile(t_fm,(len(Col),768))
            
    return ctf_bert
            
def store_ctf_bert(TableSet,arg1=None):
    
    model=BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    Fail=[]
    if arg1==None:
        ctf_dir='处理后/处理后/column_type/'
    else:
        ctf_dir='处理后/处理后/'+arg1
        
    
    bert_dir=ctf_dir
    for i in range(TableSet.table_num):
        print(i)
        if 1>0:
            W=TableSet.table_set[i]
            ctf_bert=bert_score_column_type(W,model,tokenizer,ctf_dir)
            f=open(bert_dir+W.name+'_ctf_bert.data','wb')
            pickle.dump(ctf_bert,f)
            f.close()
            #print(ctf_bert)
        else:
            Fail.append(i)
            print('fail')
    f=open(bert_dir+TableSet.name+'_fail.data','wb')
    pickle.dump(Fail,f)
    f.close()          
    
def quantity_type_bert_classification():

    f=open('../KG/KG_data/type_quantity_classified.data','rb')
    Qd=pickle.load(f)
    f.close()      
    
    f=open('type_bert_vector.data','rb')
    TB=pickle.load(f)
    f.close()
    
    model=BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    Yd=[]
    Sd=[]
    
    for q in Qd:
        print(q in TB[0])
        if q in TB[0]:
            
            loc=TB[0].index(q)
            B=TB[1][loc]
        else:
            batch=tokenizer([q],padding=True,return_tensors="pt")
            fm = model(batch['input_ids'], batch['attention_mask'], output_hidden_states=True)[1]
            B=fm.detach().numpy()[0]
            
        if Qd[q]==True:
            
            Yd.append((q,B))
            
        else:
            
            Sd.append((q,B))
            
    return Yd,Sd

def type_bert_commend():
    
    model=BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    f=open('type_bert_vector.data','rb')
    TB=pickle.load(f)
    f.close()
    
    f=open('type_list.data','rb')
    TL=pickle.load(f)
    f.close()
    
    TN=np.zeros((len(TL),768))
    TN[0:len(TB[0])]=TB[1]
    i=len(TB)
    for i in range(len(TL)):
        
        t=TL[i]
        if t not in TB[0]:
            
            #TB[0].append(t)
            batch=tokenizer([t],padding=True,return_tensors="pt")
            fm = model(batch['input_ids'], batch['attention_mask'], output_hidden_states=True)[1]
            B=fm.detach().numpy()[0]
            TN[i]=B
            
        else:
            
            loc=TB[0].index(t)
            TN[i]=TB[1][loc]
            
    NTB=(TL,TN)
    return NTB
    
def row_bert_score(WT,model,tokenizer):
    
    B={}
    S=[]
    W=[]
    for r in range(len(WT['con'])):
        
        s=''
        WC={}
        for c in range(WT['col_num']):
            
            if len(s)>512 or c>=len(WT['con'][r]):
                continue
            s=s+WT['con'][r][c]
            
            if c!=WT['col_num']-1:
                s=s+' '
        # 对句子进行分词并编码
        tokens = tokenizer.tokenize(s)
        inputs = tokenizer(s, return_tensors='pt', padding=True, truncation=True)

        outputs = model(**inputs)
        SE=outputs.last_hidden_state[:, 0, :]#.mean(dim=1)
        #print(SE.detach().numpy()[0][0],r)
        S.append(SE.detach().numpy())        
        last_hidden_states = outputs.last_hidden_state
        for c in WT['qcol']:
            
            if int(c) not in W:
                WC[c]=[]
            if len(WT['con'][r])<WT['col_num']:
                mention=WT['con'][r][-1]
                #print(mention)
            else:
                mention=WT['con'][r][int(c)]
            
            # 分词子字符串
            
            substring_tokens = tokenizer.tokenize(mention)
            substring_indices = [tokens.index(token) for token in substring_tokens if token in tokens]
            substring_vectors = last_hidden_states[0, substring_indices, :]
            if substring_vectors.size()[0]==0:
                #print('y')
                average_vector=torch.zeros((1,768))
                #print(substring_vectors.size()[0],r,substring_vectors.dim())
                # 计算所有相关token向量的平均值
            else:
                average_vector = torch.mean(substring_vectors, dim=0)
            # 打印平均向量
            SEW=average_vector.detach().numpy()
            #print(SEW[0])
        
            WC[c].append(SEW)
        W.append(WC)
    B['whole_row']=S
    B['single_word']=W
    
    return B
        
def store_row_bert_score(TS_name,arg):

    model=BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if arg==None:
        bert_dir='BERT/'
        table_dir='Data/'
    else:
        if 'null' in arg:
            r=arg.split('_')[-1]
            table_dir='处理后/处理后/'+'null_table/Rate_'+r+'/Data/'
            bert_dir='处理后/处理后/'+'null_table/Rate_'+r+'/BERT/'
        if 'sample' in arg:
            r=arg.split('_')[-1]
            table_dir='处理后/处理后/'+'sample_table/Num_'+r+'/Data/'
            bert_dir='处理后/处理后/'+'sample_table/Num_'+r+'/BERT/'
            
    with open(table_dir+TS_name+'_table.json','r',encoding='utf-8') as file:
        TableSet=json.load(file)
    os.makedirs(bert_dir,exist_ok=True)
    try:
        with open(bert_dir+TS_name+'_qcb.json','r',encoding='utf-8') as file:
            QCB=json.load(file)
    except:
        QCB={}
    j=0
    for i in TableSet:
        #print(i)
        if 1>0:
            W=TableSet[i]
            if i in QCB and QCB[i]!='fail':
                continue
            j+=1
            print(i)
            B=row_bert_score(W,model,tokenizer)
            for r in range(len(B['single_word'])):
                #print(r,B['single_word'][r].keys())
                for col in B['single_word'][r]:
                    #print(col,len((B['single_word'][r][col])))
                    B['single_word'][r][col]=B['single_word'][r][col][0].tolist()
                
                B['whole_row'][r]=B['whole_row'][r].tolist()[0]
            QCB[i]=B
            #print(ctf_bert)
        else:
            QCB[i]='fail'
            print('fail')
        if j%100==0:
            with open(bert_dir+TS_name+'_qcb.json','w') as f:
                json.dump(QCB,f)
    with open(bert_dir+TS_name+'_qcb.json','w') as f:
        json.dump(QCB,f)       
    
def header_bert_score(WT,model,tokenizer,db_name):
    
    #H=list(csv.reader(open('处理后/处理后/处理后数据集/'+db_name+'/'+WT.name+'.csv','r',encoding='utf-8')))
    H=WT['header']
    B={}    
    for c in WT['qcol']:
        
        
        h=H[int(c)]

        batch=tokenizer([h],padding=True,return_tensors="pt")
        fm = model(batch['input_ids'], batch['attention_mask'], output_hidden_states=True)[1]
        B[int(c)]=fm.detach().numpy()[0]
    
    return B    

def store_header_bert_score(TS_name):
    
    model=BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_dir='BERT/'
            
    with open('Data/'+TS_name+'_table.json','r',encoding='utf-8') as file:
        TableSet=json.load(file)
    
    try:
        with open(bert_dir+TS_name+'_hb.json','r',encoding='utf-8') as file:
            H=json.load(file)
    except:
        H={}
    for i in TableSet:
        #print(i)
        
        if 1>0:
            W=TableSet[i]
            if i in H and H[i]!='fail':
                continue
            B=header_bert_score(W,model,tokenizer,TS_name)
            for col in B:
                
                B[col]=B[col].tolist()
            H[i]=B
            #print(ctf_bert)
        else:
            H[i]='fail'
            print('fail')
    with open(bert_dir+TS_name+'_hb.json','w') as f:
        json.dump(H,f)         

if __name__=="__main__":

    store_candidate_bert_score('SemTab')