import numpy as np
from sklearn.cluster import KMeans 
import pickle
from candidate_generate import jaccard_similarity 
#from class_column_type import ColumnType
import os
from unlinkable_sample import read_new_candidate_null,read_new_tableset,read_new_candidate_sample
import json
from transformers import BertModel,BertTokenizer
from bert_score import bert_score_column_type, store_row_bert_score, store_header_bert_score



def ColumnType(t_name,name,col,can_s,bert_s,ic,qcol,total_s):
    
    C={}
    C['t_name']=t_name
    C['name']=name
    C['col']=col
    C['cs']=can_s
    C['bs']=bert_s
    C['ic']=ic
    C['qcol']=qcol
    C['ts']=total_s
    
    return C
def jaccard_similarity_type(t1, t2):
    
    set1=set(t1)
    set2=set(t2)
    
    union_size = len(set1 | set2)
    intersection_size = len(set1 & set2)
    
    
    if union_size == 0:
        return 0.0
    else:
        return intersection_size / union_size         
def Type_array(can_set):
    
    n=len(can_set)
    eer=np.zeros((n,n))
    ee=np.zeros((n,n))
    el=np.zeros((n,n))
    ll=np.zeros((n,n))
    
    for i in range(n):
        
        for j in range(n):
            if i>j:
                can0=can_set[i]
                can1=can_set[j]
                if can0['lit']==False and can1['lit']==False:
                        
                    e1=jaccard_similarity_type(can0['type'], can1['type'])
                    e2=jaccard_similarity_type(can0['rin'], can1['rin'])
                    eer[i][j]=(e1+e2)/2
                    eer[j][i]=eer[i][j]
                    
                elif can0['lit']==True and can1['lit']==True:
                    ll[i][j]=jaccard_similarity_type(can0['rin'], can1['rin'])
                    ll[j][i]=ll[i][j]                    
                        
                else:
                    
                    el[i][j]=jaccard_similarity_type(can0['rin'], can1['rin'])
                    el[j][i]=el[i][j]
            
    return ee,el,ll,eer
    
def Type_Similarity(ee,el,ll,eer,loc):
    
    m,n=np.shape(ee)
    
    #N1=n-np.size(np.argwhere(ee[loc]))
    #N2=n-np.size(np.argwhere(el[loc]))
    #N3=n-np.size(np.argwhere(ll[loc]))
    if np.max(ee)!=0:
        een=ee/np.max(ee)
    else:
        een=np.zeros((m,n))
    if np.max(el)!=0:
        
        eln=el/np.max(el)
    else:
        eln=np.zeros((m,n))
        
    if np.max(ll)!=0:
        
        lln=ll/np.max(ll)
    else:
        lln=np.zeros((m,n))
    if np.max(eer)!=0:
        
        eern=eer/np.max(eer)
    else:
        eern=np.zeros((m,n))
        
    #print(np.sum(een[loc]),np.sum(eln[loc]),np.sum(lln[loc]))
    ts=(np.sum(een[loc])+np.sum(eln[loc])+np.sum(lln[loc])+np.sum(eern[loc]))/n
    
    return ts
        
def candidate_score(can_set,col_num):
    
    can_s={}
    can_tarray={}
    for col in range(col_num):
        can_col=[]
        
        for p in can_set:
            p1=int(p.split('_')[-1])
            if p1==col:
                can_col+=can_set[p]
        ee,el,ll,eern=Type_array(can_col)
        can_tarray[col]={}
        can_tarray[col]['ts']=[ee,el,ll,eern]
        can_tarray[col]['set']=can_col
    
    #print(can_set.keys(),type(can_set))
    for pos in can_set:
        can_s[pos]=[]
        #print(pos)
        p1=int(pos.split('_')[-1])
        for c in can_set[pos]:
            #print(c)
            ls=jaccard_similarity(c['name'],c['mention'])
            loc=can_tarray[p1]['set'].index(c)
            #print(loc,can_tarray[pos[1]]['ts'][0])
            ts=Type_Similarity(can_tarray[p1]['ts'][0],can_tarray[p1]['ts'][1],can_tarray[p1]['ts'][2],can_tarray[p1]['ts'][3],loc)
            #print(can_set[pos].index(c))
            can_s[pos].append([ls,ts])
            
    return can_s

def column_type_start(can_set,can_s):
    
    Column_type={}
    can_num={}
    
    for pos in can_set:
        p1=int(pos.split('_')[-1])
        if p1 not in Column_type:
            Column_type[p1]={}
        if p1 not in can_num:
            can_num[p1]=0
        #print(pos)
        if can_set[pos]!=[] and len(can_set[pos])>1:
            Max_bert_score=max([c['bs'] for c in can_set[pos]])
        elif len(can_set[pos])==1:
            Max_bert_score=float('inf')
        else:
            Max_bert_score=0
        
        for i in range(len(can_set[pos])):
            c=can_set[pos][i]
            if c['lit']==False:
                ct=list(set(c['type']+c['rin']))
                
                for t in ct:
                    t_normal=str(t).split('/')[-1]
                    if t_normal not in Column_type[p1]:
                        Column_type[p1][t_normal]=(1-((c['bs']/Max_bert_score) if Max_bert_score!=0 else (1-c['bs'])))*(can_s[pos][i][0]+can_s[pos][i][1])
                        #print(t_normal,Column_type[pos[1]][t_normal],c['bs'])
                    else:
                        Column_type[p1][t_normal]+=(1-((c['bs']/Max_bert_score) if Max_bert_score!=0 else (1-c['bs'])))*(can_s[pos][i][0]+can_s[pos][i][1])
            else:
                
                cr=c['rin']
                for t in cr:
                    if can_s[pos][i][0]==0 or can_s[pos][i][1]==0:
                        continue
                    t_normal=str(t).split('/')[-1]
                    if t_normal not in Column_type[p1]:
                        Column_type[p1][t_normal]=(1-((c['bs']/Max_bert_score) if Max_bert_score!=0 else (c['bs'])))*(can_s[pos][i][0]+can_s[pos][i][1])
                    else:
                        Column_type[p1][t_normal]+=(1-((c['bs']/Max_bert_score) if Max_bert_score!=0 else (c['bs'])))*(can_s[pos][i][0]+can_s[pos][i][1])
            
            can_num[p1]+=len(can_s[pos])
    ct={} 
    #print(Column_type)       
    for col in Column_type:
        ct[col]={}
        if Column_type[col]=={}:
            
            continue
        
        #print(Column_type)
        M=max(list(Column_type[col].values()))
        #print(M)
        for t in Column_type[col]:
            if Column_type[col][t]!=0:
                ct[col][t]=(Column_type[col][t])/M
                
        ct[col]=sorted(ct[col].items(),key=lambda kv:(kv[1],kv[0]),reverse=True)
    #print(ct)
    return ct         
                
def column_type_process(Cot):
    
    ct={}
    for col in Cot:
        T=[i[0] for i in Cot[col]]
        ct[col]={}
        for t in T:
            #print(Tw[col][t])
            #print(Cot)
            loc=T.index(t)
            ct[col][t]=Cot[col][loc][1]
    
    cot={}    
    for col in ct:
        cot[col]={}
        if ct[col]=={}:
            continue
        T=list(ct[col].keys())
        M=max(list(ct[col].values()))
        fl=1
        if 'owl#thing' in T and ct[col]['owl#thing']==M:
            fl=0
            #loc=T.index('owl#thing')
            ct[col]['owl#thing']=0
            M=max(list(ct[col].values()))
            if 'name' in T and ct[col]['name']==M:
                #loc2=T.index('name')
                ct[col]['name']=0
                M=max(list(ct[col].values()))
                fl=2
        elif 'name' in T and ct[col]['name']==M:
            ct[col]['name']=0
            M=max(list(ct[col].values()))
            fl=3
            #print(M,col)
        M=max(list(ct[col].values()))
        #print(M,fl,ct[col].values(),ct)
        for t in ct[col]:
            if fl==0 and t=='owl#thing':
                cot[col][t]=1
            elif fl==2 and t=='owl#thing':
                cot[col][t]=1
            elif fl==3 and t=='name':
                cot[col][t]=1
            else:
                #print(fl,col,M)
                cot[col][t]=ct[col][t]/M
                #print(fl,cot[col][t],M)
                
        
        cot[col]=sorted(cot[col].items(),key=lambda kv:(kv[1],kv[0]),reverse=True)
    
    return cot
                            
def column_type_filter(WT,ctf_can,ctf_bcan,alpha,ic):
    
    CT=initial_column_type(WT,ctf_can, ctf_bcan,alpha,ic)
    #print(CT)
    New_CT={}
    for col in CT:
        New_CT[col]=[]
        if len(CT[col])==0:
            continue
        elif len(CT[col])==1:
            New_CT[col]=CT[col]
            continue
        #print(CT[col])
        type_score=(np.array([i['ts'] for i in CT[col]])).reshape(-1,1)
        #print(type_score)
        k=2
        kmodel = KMeans(n_clusters = k)
        kmodel.fit(type_score)
        #print(kmodel.labels_)
        flag=kmodel.labels_[0]
        
        
        for f in range(len(CT[col])):
        
            if kmodel.labels_[f]!=flag:
                if CT[col][0]['ts']>CT[col][f]['ts']:
                    
                    ff=kmodel.labels_[0]
                else:
                    ff=kmodel.labels_[f]
        for f in range(len(CT[col])):
            
            if kmodel.labels_[f]==ff:
                
                New_CT[col].append(CT[col][f])
                
    return CT,New_CT

def column_type_main(WT,Can_set,arg=None):
    
    #Can_set=read_candidate(WT)
    '''
    if arg!=None:
        
        if 'null' in arg:
            
            Can_set=read_new_candidate_null(WT, Can_set)
            
        elif 'sample' in arg:
            
            n=int(arg.split('_')[-1])
            Can_set=read_new_candidate_sample(WT, Can_set, n)
    '''        
    #print(Can_set.keys())
    can_score=candidate_score(Can_set, WT['col_num'])
    
    column_candidate=column_type_start(Can_set,can_score)
    column_type=column_type_process(column_candidate)
    Column_type={}
    for col in column_type:
        if len(column_type[col])<10:
            
            Column_type[col]=column_type[col]
        else:
            Column_type[col]=column_type[col][0:10]
    #print(column_type)
    #Column_type=column_type_filter(column_type)
    return Column_type

def store_column_type_candidate(TS_name,recommend,arg1=None):
    if arg1==None:
        col_type_dir='ColumType/'
        with open('Data/'+TS_name+'_table.json','r',encoding='utf-8') as file:
            TableSet=json.load(file)
        with open('Candidate/'+TS_name+'_fcs.json','r',encoding='utf-8') as file:
            C=json.load(file)
    else:
        if 'null' in arg1:
            #print(args)
            r=arg1.split('_')[-1]
            col_type_dir='Data/'+'null_table/Rate_'+r+'/column_type/'
            os.makedirs(col_type_dir,exist_ok=True)
            con_dir='Data/'+'null_table/Rate_'+r+'/table_content/'
            TableSet=read_new_tableset(TableSet, con_dir)
        else:
            r=arg1.split('_')[-1]
            col_type_dir='Data/'+'sample_table/Num_'+r+'/column_type/'
            os.makedirs(col_type_dir,exist_ok=True)
            con_dir='Data/'+'sample_table/Num_'+r+'/table_content/'
            TableSet=read_new_tableset(TableSet, con_dir)
    if recommend==False:
        #Fail=[]
        T={}
        for i in TableSet:
            print(i)
            try:
                W=TableSet[i]
                #M=TableSet.table_name[i]
                #print(W.sam_con)
                CT=column_type_main(W,C,arg1)
                #print(CT)
                T[i]=CT
            except:
                T[i]='Fail'
        
        with open('ColumnType/'+TS_name+'_ctf_can.json','w') as f:
            json.dump(T,f)
    else:
        with open('ColumnType/'+TS_name+'_ctf_can.json','r',encoding='utf-8') as file:
            T=json.load(file)
        for i in TableSet:
            try:
                if T[i]=='Fail':
                    print(i)
                    W=TableSet[i]
                    #M=TableSet.table_name[i]
                    CT=column_type_main(W,arg1)
                    T[i]=CT
            except:
                T[i]='Fail'
        with open('ColumnType/'+TS_name+'_ctf_can.json','w') as f:
            json.dump(T,f)

        
def initial_column_type(WT,ctf_can,ctf_bcan,alpha,ic):
    
    CT={}
    
    for col in ctf_can:
        CT[col]=[]
        
        if len(ctf_can[col])==0:
            
            continue
        
        M=max([c[1] for c in ctf_bcan[col]])
        for t in ctf_can[col]:
            
            ct=ColumnType(WT['id'],t[0],col,t[1],1,1,False,1)
            ct['bs']=1-ct['bs']/M
            ct['ic']=ic[t[0]]
            ct['ts']=(ct['ic']**(1/2))*(alpha*ct['cs']+(1-alpha)*ct['bs'])
            CT[col].append(ct)
            
    return CT
    
    
def store_column_type(TS_name,arg1=None):
    if arg1==None:
        
        table_dir='Data/'
        can_dir='Candidate/'
        col_type_dir='ColumnType/'
        
    else:
        if 'null' in arg1:
            #print(args)
            r=arg1.split('_')[-1]
            table_dir='Data/'+'null_table/Rate_'+r+'/Data/'
            can_dir='Data/'+'null_table/Rate_'+r+'/Candidate/'
            col_type_dir='Data/'+'null_table/Rate_'+r+'/ColumnType/'
            #con_dir='Data/'+'null_table/Rate_'+r+'/table_content/'
            #TableSet=read_new_tableset(TS_name, con_dir)
        else:
            r=arg1.split('_')[-1]
            table_dir='Data/'+'sample_table/Num_'+r+'/Data/'
            can_dir='Data/'+'sample_table/Num_'+r+'/Candidate/'
            col_type_dir='Data/'+'sample_table/Num_'+r+'/ColumnType/'
    
    os.makedirs(col_type_dir,exist_ok=True)

    with open(table_dir+TS_name+'_table.json','r',encoding='utf-8') as file:
        TableSet=json.load(file)
    
    with open(can_dir+TS_name+'_fcs.json','r',encoding='utf-8') as file:
        C=json.load(file)   
        
    try:
        with open(col_type_dir+TS_name+'_ctf.json','r',encoding='utf-8') as file:
            CTF=json.load(file)
    except:
        #PC={}
        CTF={}
    f=open('ic_type.data','rb')
    ic=pickle.load(f)
    f.close()
    model=BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    j=0
    #Fail=[]
    #CTF={}
    for i in TableSet:
        j+=1
        #if j>3:
        #    break
        if i in CTF and CTF[i]!='Fail':
            continue
        print(i,j)
        if 1>0:
            W=TableSet[i]
            CT=column_type_main(W,C[i],arg1)
            #print(CT)
            CTB=bert_score_column_type(W,model,tokenizer,CT)
            _,NCT=column_type_filter(W,CT,CTB,0.3,ic)
            CTF[i]=NCT
            
        else:
            CTF[i]='Fail'
            print('fail')
        if j%100==0:
            with open(col_type_dir+TS_name+'_ctf.json','w') as f:
                json.dump(CTF,f)
    with open(col_type_dir+TS_name+'_ctf.json','w') as f:
        json.dump(CTF,f)


        
if __name__=="__main__":
    
    TS_name='T2DC'
    arg1=None
    #store_column_type(TS_name,arg1)
    #store_row_bert_score(TS_name, arg1)
    
    for r in ['0.3','0.4','0.5']:
        store_column_type(TS_name,'null_'+r)
        store_row_bert_score(TS_name, 'null_'+r)
    
    #store_header_bert_score(TS_name)




