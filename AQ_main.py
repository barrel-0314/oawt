import json
#from transformers import BertModel,BertTokenizer
import csv
from aggregate_query_util import extract_query_keyword
from fuzzywuzzy import process
#import torch
import numpy as np
import time
from tqdm import tqdm
from CI import sample_table_main,calculate_op_result,table_main,confidence_interval_sample,confidence_interval_noise,pro_cdf
import pickle
import threading

def exact_match_column(K,CT,TS_name,table_name):
    
    CT={}
    return match_column(K, CT, TS_name, table_name)
def match_column(K,CT,TS_name,table_name):
    
    k_col={}
    flag=True
    for k in K:
        k_col[k]=[]
        #print(CT)
        H=list(csv.reader(open('处理后/处理后/处理后数据集/'+TS_name+'/'+table_name+'.csv','r',encoding='utf-8')))[0]
        for h in range(len(H)):
            
            #s=[j[0] for j in H if k.lower() in j.lower()]
            if k.lower() in H[h].lower():
                k_col[k].append(h)
        
        if len(k_col[k])==0:

            for col in CT:
                
                s=[j[0] for j in process.extract(k,CT[col]) if j[1]>90]
                if len(s)>0:
                    k_col[k].append(int(col))
            
            
        if len(k_col[k])==0:
            flag=False
            break
    
    if flag:
        return flag,k_col
    else:
        return flag,-1
    
        
def match_table(K_match,union,CT,TS_name):
    
    #K=[k1,k2,k3]
    #K_match=[k for k in K if k!='']
    Match={}
    for t in union:
        #print(CT[t])
        if t not in CT:
            continue
        flag,m=match_column(K_match,CT[t],TS_name,t)
        if flag:
            #print('u',t,flag,m)
            Match[t]=m
            
    return Match
def find_first_table(K,TS_name,CT):
    
    with open('Data/'+TS_name+'_table.json','r',encoding='utf-8') as file:
        TableSet=json.load(file)
        
    flag=False
    m=[]   
    
    for i in TableSet:
        CT[i]={}
        #W=TableSet[i]
        flag,m=match_column(K,CT[i],TS_name,i)
        #print(i,'tt')
        if flag:
            #print(flag,m)
            break
    if flag:
        return i,m
    else:
        
        return [],m
    
def timer(timeout):
    """计时器函数，当超时后设置标志变量"""
    
    global time_is_up
    time.sleep(timeout)
    time_is_up = True
    print("Time is up! Exiting...")
    
def Aggregate_Query(TS_name,Pro,error,I):
    
    with open('Query/'+TS_name+'_aq.json','r',encoding='utf-8') as f:
        Q=json.load(f)
    with open('Biased/'+TS_name+'_union.json','r',encoding='utf-8') as file:
        U=json.load(file)
    '''
    with open('Query/'+TS_name+'_col_name.json','r',encoding='utf-8') as f:
        CT=json.load(f) 
    
    with open('Query/'+TS_name+'_ql'+'.json','r',encoding='utf-8') as file:
        QL=json.load(file)  
    
    with open('Query/'+TS_name+'_err.json','r',encoding='utf-8') as f:
        Err=json.load(f)   
    
    with open('Query/'+TS_name+'_res_ci.json','r',encoding='utf-8') as f:
        R=json.load(f)   
    
    '''
    '''
    with open('label_information/'+TS_name+'_ctf.data','rb') as file:
        CTA=pickle.load(file)
    with open('label_information/'+TS_name+'_label_new.json','r',encoding='utf-8') as file:
        L=json.load(file)
    with open('Query/'+TS_name+'_res.json','r',encoding='utf-8') as f:
        T=json.load(f)
    for t in CT:
        #print(t)
        C=list(csv.reader(open('处理后/处理后/处理后数据集/'+TS_name+'/'+t+'.csv','r',encoding='utf-8')))[0]

        for col in CT[t]:
            CT[t][col]=[]
            if t in CTA:
                if col in CTA[t]:
                    CT[t][col].extend(CTA[t][col])            
            if t in T:
                if col in T[t]:
                    CT[t][col].extend(T[t][col])
            if t in L:
                if col in L[t]:
                    CT[t][col].extend(L[t][col])
            CT[t][col].append(C[int(col)])
            CT[t][col]=list(set(CT[t][col]))
    #return CT
    '''
    R={} 
    M={}
    CT={}
    j=0
    #error=0.05
    pro=1-236/253
    del_q=[]
    for q in Q:
        op=Q[q]['SELECT'].split('(')[0]
        if op=='MAX' or op=='MIN':
        #if op!='AVG':   
        #if q!='Query 85':
            del_q.append(q)
        #if q not in QL:
        #    del_q.append(q)
    for d in del_q:
        if d in Q:
            del Q[d]
    print(len(Q))
    st=time.time()
    for q in tqdm(Q,desc='query index',total=len(Q)):
        #if q not in QL:
        #    continue
        j+=1
        if j>5:
        #    continue
        #if q.split()!=56:
            break
        M[q]={}
        query=Q[q]
        k1,k2,k3=extract_query_keyword(query)
        if k3=='countries':
            k3='country'
        #print(k1,k2,k3)
        K=[k1,k2,k3]
        K_match=[k for k in K if k!='']
        t,m=find_first_table(K_match,TS_name,CT)
        
        if isinstance(t, list):
            #print(k1,k2,k3)
            R[q]=0
            time.sleep(1)
            #print(T[q])
            continue
        #M[q][t]=m
        if t in U:
            M[q]=match_table(K_match,U[t],CT,TS_name)
        M[q][t]=m
        #print(len(U[t]))
        #U[t]=U[t][0:50]
        #U[t]=(U[t]*20)[0:500]
        ST=[]
        ite=0
        P=0
        SR=[]
        TR={}
        
        P0=100
        while P<Pro and ite<I:
            if abs(P-P0)<0.001:
                break
            SN,SR,TR,ST=sample_table_main(M[q],K,TS_name,query,70,True,ST,SR,TR)
            ER,NA,op=calculate_op_result(query,SR,TR,SN,K)
            #print(len(SR[0]),len(ST[0]),op,K)
            mu1,sig1=confidence_interval_sample(ER, TR, SN, NA, op, SR, error)
            #mu2=0
            #sig2=0
            mu2,sig2=confidence_interval_noise(ER, TR, SN, NA, op, SR, error, pro)
            mu=mu1+mu2
            #print(mu1,mu2,sig1,sig2)
            if np.isnan(sig2):
                sig2=0
            sigma=(sig1**2+sig2**2)**(1/2)
            P0=P
            P=pro_cdf(mu, sigma, error, ER, SN)
            #print(P)
            ite+=1
        #if len(SR)==0:
        #    print(q.split(' '))[-1]
        #SRR[q]=SR
        R[q]=ER
        time.sleep(1)
            
        #if j>0:
        #    break
    ed=time.time()
    if j!=0:
        
        print((ed-st),'time_'+TS_name)
        print(Pro,error)
    
    '''
    with open('Query/'+TS_name+'_res_ci_error'+str(error)+'.json','w') as f:
        json.dump(R,f)
    '''
    return R

if __name__=="__main__":
    
    TS=['SemTab','T2D','wiki','T2DC','TUS','TGPT']
    
    for TS_name in TS[5:6]:
        R=Aggregate_Query(TS_name,0.95,0.05,20)
    
    '''
    E={}
    #error=[0.1,0.15,0.2,0.25]
    #error=[0.15]
    Pro=[0.9]
    #Pro=[0.95]
    
    for TS_name in TS[0:1]:
        E[TS_name]={}
        #for e in error:
        for e in Pro:
            #_,E[TS_name][e]=Aggregate_Query(TS_name,0.95,e,100)
            _,E[TS_name][e]=Aggregate_Query(TS_name,e,0.05,100)
    #p=1-236/253
    '''