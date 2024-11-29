import torch
from transformers import BertModel,BertTokenizer
import json
import os
import numpy as np
import torch.nn.functional as F


def mention_can_diff(men,can,model,tokenizer):
    
    batch=tokenizer([men],padding=True,return_tensors="pt")
    fm = model(batch['input_ids'], batch['attention_mask'], output_hidden_states=True)[1]
    batch=tokenizer([can],padding=True,return_tensors="pt")
    fc = model(batch['input_ids'], batch['attention_mask'], output_hidden_states=True)[1]
    N=F.cosine_similarity(fc, fm).item()
    #dis=torch.sub(fc,fm)
    #N=torch.norm(dis)
    
    #print(N)
    return float(N)

def construct_graph(T,qt,qcol,model,tokenizer,g):
    
    core=int(T['core'])
    mention_node_core=[j[core] if core<len(j) else j[-1] for j in T['con']]
    mention_node_q=[j[qcol] if qcol<len(j) else j[-1] for j in T['con']]
    
    mention_node=mention_node_core+mention_node_q
    
    type_node=qt
    
    m=len(mention_node)
    t=len(type_node)
    S=torch.randn(m+t,m+t)
    #all connected
    #S0=torch.zeros(m+t,m+t)
    for i in range(m):
        
        for j in range(i-1):
            
            S[i][j]=mention_can_diff(mention_node[i], mention_node[j], model, tokenizer)
            S[j][i]=S[i][j]
    
    S=S*(1/S.max())
    for i in range(m+t):
        
        S[i][i]=1
    S0=torch.diag(S.sum(dim=1))
        
    G=torch.eye(2,2)
    G[0][0]=g
    C=torch.ones(m+t,2)
    for i in range(m):
        C[i,:]=torch.tensor([0.55,0.45])
    for i in range(t):
        
        C[m+i,:]=C[m+i,:]*0.5
    #print(S0-S)    
    return S,S0,G,C,mention_node,type_node

def update_community(S,S0,G,C,alpha,beta):
    
    delta=1
    n=C.size(0)
    ite=0
    CI=C
    while delta>10**(-5):
        
        U=(torch.matmul(S,C)+alpha*torch.ones(n,2))/(torch.matmul(S0,C)+alpha*torch.matmul(C,torch.ones(2,2))+beta*(torch.matmul(C,G)))
        
        C0=torch.zeros(n,2)
        for i in range(n):
            
            for j in range(2):
                
                if U[i][j]>0:
                    C0[i][j]=C[i][j]*(U[i][j]**(0.5))
                else:
                    C0[i][j]=C[i][j]
                
                #if np.isnan(C0[i][j]):
                    
                #    print(C[i][j],U[i][j])
                
        delta=((C-C0)**2).sum()
        #print(delta)
        if delta>100:
            C=CI
            #ite=[]
        else:
            ite+=1
            C=C0
        if ite>1000:
            break
    #L1=torch.matmul(torch.matmul(C.t(),S0-S),C).diagonal().sum()
    #L2=alpha*(torch.norm(torch.matmul(C,torch.ones(2))-torch.ones(n))**2)
    #L3=beta*(torch.matmul(torch.matmul(G,C.t()),C)).diagonal().sum()
    
    #print(L1,L2,L3,ite)
    return C    

def identify_result(S,C,qt):
    
    bia=False
    non_correct=False
    
    Cmain=torch.nonzero(C[:,0]>C[:,1],as_tuple=True)[0]
    Cother=torch.nonzero(C[:,0]<=C[:,1],as_tuple=True)[0]
    
    nmain=Cmain.size(0)
    nother=Cother.size(0)
    Gmain=S[Cmain,Cmain].sum()/nmain
    Gother=S[Cother,Cother].sum()/nother
    
    if Gmain<Gother:
        
        bia=True
        #print(Gmain,Gother)
        return bia, non_correct
    else:
        
        m=C.size(0)-len(qt)
        #print(type(torch.where(Cmain<m)))
        men1=torch.where(Cmain<m)[0].size(0)
        men2=torch.where(Cother<m)[0].size(0)
        if men1<men2:
            bia=True
            #print(men1,men2)
            return bia,non_correct
        else:
            ty1=torch.where(Cmain>=m)[0].size(0)
            if ty1==0:
                #print(Cother)
                non_correct=True
            return bia,non_correct    
    
    

def refine_qtype(TS_name,arg1,model,tokenizer):
    
    if arg1==None:
        
        table_dir='Data/'
        res_dir='Result/'+TS_name+'_res.json'
        bia_dir='Biased/'
        #res_dir='../../aggregate/Result/'
    elif 'null' in arg1:
        
        r=arg1.split('_')[-1]
        table_dir='处理后/处理后/'+'null_table/Rate_'+r+'/Data/'
        res_dir='../Llama/TableLlama-main/output_data/'+'noise/'+TS_name+'_res_'+r+'.json'
        bia_dir='处理后/处理后/noise_table/Rate_'+r+'/Biased/'
        #res_dir='../../aggregate/处理后/处理后/'+'null_table/Rate_'+r+'/Result/'
        
    else:
        
        r=arg1.split('_')[-1]
        table_dir='处理后/处理后/'+'sample_table/Num_'+r+'/Data/'
        res_dir='../Llama/TableLlama-main/output_data/'+'sample/'+TS_name+'_res_'+r+'.json'
        bia_dir='处理后/处理后/sample_table/Num_'+r+'/Biased/'
        #res_dir='../../aggregate/处理后/处理后/'+'sample_table/Num_'+r+'/Result/'
        
    os.makedirs(bia_dir,exist_ok=True)

    with open(table_dir+TS_name+'_table.json','r',encoding='utf-8') as file:
        TableSet=json.load(file)
        
    with open(res_dir,'r',encoding='utf-8') as file:
        QT=json.load(file)
    for k in range(5):
        try:
            with open(bia_dir+TS_name+'_bia.json','r',encoding='utf-8') as f:
                R=json.load(f)  
        except:
            R={}
            R['bia']={}
            R['non']={}
        
        j=0
        g=0.25
        alpha=0.5
        beta=0.1
        #R={}
        for i in TableSet:
            
            j+=1
            #print(j)
            W=TableSet[i]
            if 'core' not in W:
                W['core']=0
            if i not in QT:
                continue
            #print(QT[i])
            '''
            if i in R['bia']:
                for qcol in R['bia'][i]:
                    #print(qcol)
                    if not(R['bia'][i][qcol]) and not (R['non'][i][qcol]):
                        B[i]={}
                        N[i]={}
                        B[i][qcol]=R['bia'][i][qcol]
                        N[i][qcol]=R['non'][i][qcol]
                        continue
            '''
            if j%100==0:
                print(j)
            
            for qcol in QT[i]:
                #print(qcol,'q')
                try:
                    if not(R['bia'][i][qcol]) and not (R['non'][i][qcol]):
                        
                        continue
                except:
                    if i not in R['bia']:
                        R['bia'][i]={}
                        R['non'][i]={}
                        
                S,S0,G,C,mn,tn=construct_graph(W,QT[i][qcol],int(qcol),model,tokenizer,g)
                C=update_community(S,S0,G,C,alpha,beta)
                cm,co=identify_result(S,C,QT[i][qcol])
                bia,nco=identify_result(S,C,QT[i][qcol])
                #print(bia,nco)
                #B[i]={}
                #N[i]={}
                #B[i][qcol]={}
                #N[i][qcol]={}
                
                R['bia'][i][qcol]=bia
                R['non'][i][qcol]=nco
            #if j>2:
                
            #    break
            #print(B[i].keys(),'q')
        #R['bia']=B
        #R['non']=N
        
        with open(bia_dir+TS_name+'_bia.json','w') as f:
            json.dump(R,f)
        
        #print(R['bia'],R['non'])
        n=0
        for i in R['bia']:
            for qcol in R['bia'][i]:
                if not(R['bia'][i][qcol]) and not (R['non'][i][qcol]):
                    n+=1
        print(n)
    return R
        
if __name__=="__main__":
    
    model=BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    TS=['SemTab','T2D','wiki','T2DC']
    
    for TS_name in TS[1:-1]:
        #TS_name='T2DC'
        arg1=None
        R=refine_qtype(TS_name,arg1,model,tokenizer)
    