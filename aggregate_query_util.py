import json
import pickle
from fuzzywuzzy import process
#import torch
#import torch.nn.functional as F
import os
from sklearn.cluster import KMeans 
import numpy as np

def extract_query_keyword(query):
    
    target=query['SELECT'].split('(')[-1]
    target=target.strip(')')
    tar=target.split('.')
    if len(tar)==2:
        key1=tar[0]
        key2=tar[-1]
    else:
        key1=target
        key2=''
    if '=' in query['WHERE'] and '.' in query['WHERE']:
        s1=query['WHERE'].split('=')[0]
        key3=s1.split('.')[-1]
        #if key3==key1 or key3==key2:
        #    key3=''
    else:
        key3=''
        
    return key1,key2,key3
def string_fuzzy_match(str1,str_list):
    
    s=process.extract(str1, str_list,limit=1)[0]
    
    return s
    
    
def vector_similarity_match(vec1,vec_list):
    
    #vec1 = vec1.float()
    #vec_list = vec_list.float()
    #print(type(vec1),type(vec_list))
    #print(vec1.shape,vec_list.shape)
    similarities = F.cosine_similarity(vec1, vec_list, dim=2).squeeze()
    #print(similarities.shape)
    max_similarity, max_index = torch.max(similarities, dim=0)
    #print(max_similarity,max_index)
    return max_similarity, max_index
        
    
def extract_type(TS_name,TableSet):
    
    col_dir='../Llama/TableLlama-main/output_data/'
    q_dir='Result/'
    col_ty_dir='ColumnType/'
    with open(col_dir+TS_name+'_col_res.json','r',encoding='utf-8') as file:
        Col=json.load(file)
        
    with open(q_dir+TS_name+'_res.json','r',encoding='utf-8') as file:
        QT=json.load(file)
    with open(col_ty_dir+TS_name+'_ctf.json','r',encoding='utf-8') as file:
        L=json.load(file)
    T={}
    
    for i in TableSet:
        
        W=TableSet[i]
        T[i]={}
        print(i)
        for j in range(W['col_num']):
            
            if W['qflag'][j]==0:
                #print(Col[i].keys())
                print(j)
                if str(j) not in L[i]:
                    ct=[]
                else:
                    ct=[t['name'] for t in L[i][str(j)]]
                T[i][j]=Col[i][str(j)]+ct
                new_list = ['location' if x == 'location.location' else x for x in T[i][j]]
                new_list=list(set([x.lower() for x in new_list]))
                if 'populationplace' in new_list:
                    new_list.remove('populationplace')
                if 'populatedplace' in new_list:
                    new_list.remove('populatedplace')
                T[i][j]=new_list
                
            else:
                if str(j) in QT[i]:
                    T[i][j]=[q.lower() for q in QT[i][str(j)]]
    
    with open('Query/'+TS_name+'_col_name.json','w') as f:
        json.dump(T,f)         
    return T
def store_table_union_result(TS_name):
    
    Union={}
    path='pylon-main/pylon-main/wte_cl/ssd/congtj/'+TS_name+'_metaspace/pylon/wte_cl/results/'+TS_name+'/420_wte_cl_epoch_9_sample_-1_lsh_0.7_topk_100/'
    L=os.listdir(path)
    
    for l in L:
        
        if l=='log.txt':
            continue
        #U={}
        with open(path+l, 'r') as file:
        
            for line in file:
                
                if 'Query table' in line:
                    
                    t=line.split('Query table: ')[-1]
                    t=t.strip('\n')
                    Union[t]=[]
                    
                elif 'Candidate table' in line:
                    
                    u=line.split('Candidate table: ')[-1]
                    u=u.strip('\n')
                    #Union[t][u]
                
                elif 'Candidate score' in line:
                    
                    s=line.split('Candidate score: ')[-1]
                    s=s.strip('\n')
                    
                    if float(s)<0.8:
                        break
                    Union[t].append(u)
                    
                    #U[u]=float(s)
        '''
        uu=list(U.items())
        type_score=(np.array([i[1] for i in uu])).reshape(-1,1)
        #print(type_score)
        k=2
        kmodel = KMeans(n_clusters = k)
        kmodel.fit(type_score)
        #print(kmodel.labels_)
        flag=kmodel.labels_[0]
        
        #print(QT_score,kmodel.labels_)
        for f in range(len(uu)):
        
            if kmodel.labels_[f]!=flag:
                if uu[0][1]>uu[f][1]:
                    
                    ff=kmodel.labels_[0]
                else:
                    ff=kmodel.labels_[f]
            
        for f in range(len(uu)):
            
            try:
                if kmodel.labels_[f]==ff:
                    
                    Union[t].append(uu[f][0])
            except:
                Union[t].append(uu[f][0])
        '''  
    with open('Biased/'+TS_name+'_union.json','w') as f:
        json.dump(Union,f)         
    return Union
def relative_error(measured_value, true_value):
    
    if true_value==0:
        error=0
    else:
        
        error = abs((measured_value - true_value) / true_value)
    #if error>1000:
    #    print(measured_value,true_value)
    return error
def result_metric(TS_name,met):
    
    with open('Query/'+TS_name+'_aq.json','r',encoding='utf-8') as f:
        Q=json.load(f)
    with open('Query/'+TS_name+'_res.json','r',encoding='utf-8') as file:
        Res=json.load(file)
    
    with open('Query/'+TS_name+'_'+met+'.json','r',encoding='utf-8') as file:
        R=json.load(file)
    
    with open('Query/'+TS_name+'_ql'+'.json','r',encoding='utf-8') as file:
        QL=json.load(file)    
    
    #with open('Query/'+TS_name+'_match.json','r',encoding='utf-8') as file:
    #    M=json.load(file)
    #E=np.zeros(len(R))
    O=['COUNT','SUM','AVG']
    #i=0
    Err={}
    err={}
    for o in O:
        Err[o]=[]
    i=0
    #QL=[]
    thr=0.7
    #print(len(R))
    for q in R:
        o=Q[q]['SELECT'].split('(')[0]
        #print(q)
        r=relative_error(R[q], Res[q])
        #print(r,type(q),q in QL)
        #Err[o].append(r)
        if q in QL:
        #if r<thr:
            #print('t')
            Err[o].append(r)
            #QL.append(q)
            #Err[o].append(r)
            err[q]=r
        '''
        if r>thr:
            print(R[q],Res[q])
            i+=1
        '''
        
            
    E={}
    print(i)
    for o in O:
        #print(Err[o])
        E[o]=np.mean(np.array(Err[o]))
        #print(len(Err[o]))
    return E,err,R,Res,QL
    
if __name__=="__main__":
    
    TS=['SemTab','T2D','wiki','T2DC']
    met='res_ci'
    E={}
    error=[0.05,0.1,0.15,0.2,0.25]
    Pro=[0.75,0.80,0.85,0.90,0.95]
    for TS_name in TS[0:1]:
        #TS_name='T2DC'
        #E[TS_name]={}
        '''
        with open('Data/'+TS_name+'_table.json','r',encoding='utf-8') as file:
            TableSet=json.load(file)
        T=extract_type(TS_name,TableSet)
        '''
        #for e in error:
        for e in Pro:
            met1=met+'_error'+str(e)
            e,res,_,_,_=result_metric(TS_name,met1)
            #for i in e:
            
            with open('Query/'+TS_name+'_err_'+met1+'.json','w') as f:
                json.dump(res,f)
            '''
            with open('Query/'+TS_name+'_ql.json','w') as f:
                json.dump(ql,f)
            '''
            
            E[TS_name]=e
            print(e,met1)
            print(np.mean(np.array(list(res.values()))))
            #print(e)