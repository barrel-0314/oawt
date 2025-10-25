import numpy as np
import networkx as nx
from HMM_graph import construct_graph
from initial_graph_input import initial_type_node,read_type_information,read_new_QCT,store_quantity_candidate
from sklearn.cluster import KMeans 
import os
import json

def PR_on_graph_list(G,WT):
    
    QT_score={}
    
    for c in G:
        
        g=G[c]
        QT_score[c]={}
        try:
            PR=nx.pagerank(g,weight='w',max_iter=1000)
            for pr in PR:
                #print(pr,g.nodes[pr]['nt'],g.nodes[pr]['nt']=='quantityType' or 'quantityHeader')
                if g.nodes[pr]['nt']=='quantityType' or g.nodes[pr]['nt']=='quantityHeader':
                    #print(pr,'pr')
                    #n=pr[-1]
                    QT_score[c][pr[2]]=PR[pr]
                    #print(QT_score[c])
        except:
            #print(c)
            node =[node for node, attrs in g.nodes(data=True) if attrs['nt']=='quantityType' or attrs['nt']=='quantityHeader']
            for n in node:
                node_w=0
                nei=list(g.predecessors(n))
                #print(len(nei))
                for nn in nei:
                    #if np.isnan(g.edges[nn,n]['w']):
                    #    print(g.edges[nn,n]['w'],nn,n)
                    node_w+=g.edges[nn,n]['w']
                    #print(node_w)
                    #print(node_w)
                QT_score[c][n[2]]=node_w/len(nei)
        #sorted_item=sorted(QT_score[c].items(),key=lambda item: item[1],reverse=True)
        #QT_score[c]=dict(sorted_item)
        
    return QT_score
        
def quantity_column_type_identification(WT,NCS,QT,QB,ic,yd,sd,TS_name,ctf,qcb,qct,col,hb,arg1):    
    
    qct=read_new_QCT(qct)
    ctf,ctfq,Col_e,Col_q=initial_type_node(WT, NCS,QT, QB, ic, TS_name, yd, sd,ctf,qcb,qct,col,hb,arg1)
    
    GL=construct_graph(NCS, WT, ctf, ctfq, Col_e, Col_q)
    QT_score=PR_on_graph_list(GL, WT)
    #print(QT_score)
    QT=KMeans_QT_extraction(QT_score)
    return QT_score,QT

def KMeans_QT_extraction(QT_score):
    
    QT={}
    for col in QT_score:
        if len(QT_score[col])==1:
            QT[col]=list(QT_score[col].keys())
            continue
        else:
            QT[col]=[]
        qt=list(QT_score[col].items())
        #rint(qt)
        type_score=(np.array([i[1] for i in qt])).reshape(-1,1)
        #print(type_score)
        k=2
        kmodel = KMeans(n_clusters = k)
        kmodel.fit(type_score)
        #print(kmodel.labels_)
        flag=kmodel.labels_[0]
        
        #print(QT_score,kmodel.labels_)
        for f in range(len(QT_score[col])):
        
            if kmodel.labels_[f]!=flag:
                if qt[0][1]>qt[f][1]:
                    
                    ff=kmodel.labels_[0]
                else:
                    ff=kmodel.labels_[f]
            
        for f in range(len(qt)):
            
            try:
                if kmodel.labels_[f]==ff:
                    
                    QT[col].append(qt[f][0])
            except:
                QT[col].append(qt[f][0])
                
    return QT
    
def store_quanitity_column_type_result(TS_name,arg1):
    if arg1==None:
        table_dir='Data/'
        can_dir='Candidate/'
        col_type_dir='ColumnType/'
        bert_dir='BERT/'
        res_dir='Result/'
    else:
        if 'null' in arg1:
            #print(args)
            r=arg1.split('_')[-1]
            table_dir='Data/'+'null_table/Rate_'+r+'/Data/'
            can_dir='Data/'+'null_table/Rate_'+r+'/Candidate/'
            col_type_dir='Data/'+'null_table/Rate_'+r+'/ColumnType/'
            bert_dir='Data/'+'null_table/Rate_'+r+'/BERT/'
            res_dir='Data/'+'null_table/Rate_'+r+'/Result/'
        else:
            #print(args)
            r=arg1.split('_')[-1]
            table_dir='Data/'+'sample_table/Num_'+r+'/Data/'
            can_dir='Data/'+'sample_table/Num_'+r+'/Candidate/'
            col_type_dir='Data/'+'sample_table/Num_'+r+'/ColumnType/'
            bert_dir='Data/'+'sample_table/Num_'+r+'/BERT/'
            res_dir='Data/'+'sample_table/Num_'+r+'/Result/'
    with open(table_dir+TS_name+'_table.json','r',encoding='utf-8') as file:
        TableSet=json.load(file)
    with open(can_dir+TS_name+'_pcan.json','r',encoding='utf-8') as file:
        NCS=json.load(file)
    with open(col_type_dir+TS_name+'_qct.json','r',encoding='utf-8') as file:
        QCT=json.load(file)
    with open(can_dir+TS_name+'_col.json','r',encoding='utf-8') as file:
        Col=json.load(file)
    os.makedirs(res_dir,exist_ok=True)
    
    with open(col_type_dir+TS_name+'_ctf.json','r',encoding='utf-8') as file:
        CTF=json.load(file)
    with open(bert_dir+TS_name+'_qcb.json','r',encoding='utf-8') as file:
        QCB=json.load(file)
    with open('BERT/'+TS_name+'_hb.json','r',encoding='utf-8') as file:
        HB=json.load(file)
    #CTF,QCB,HB=read_column_type_whole(TS_name)

    QT,QB,ic,yd,sd=read_type_information()
    #td,wd=read_index_info()
    try:
        with open(res_dir+TS_name+'_qs.json','r',encoding='utf-8') as file:
            QS=json.load(file)
        with open(res_dir+TS_name+'_qt.json','r',encoding='utf-8') as file:
            Q=json.load(file)
    except:
        Q={}
        QS={}
    
    for i in TableSet:
        
        
        if 1>0:
            if i in Q and Q[i]!='Fail':
                #print(Q[i])
                continue
            
            W=TableSet[i]
            print(i)
            if 'core' not in W:
                W['core']=0
            #print(QCT[i])
            QS[i],Q[i]=quantity_column_type_identification(W,NCS[i],QT,QB,ic,yd,sd,TS_name,CTF[i],QCB[i],QCT[i],Col[i],HB[i],arg1)
            
        else:
            QS[i]='Fail'
            Q[i]='Fail'
            print('fail')
    
    with open(res_dir+TS_name+'_qs.json','w') as f:
        json.dump(QS,f)
    with open(res_dir+TS_name+'_qt.json','w') as f:
        json.dump(Q,f)
    
if __name__=="__main__":

    TS_name='SemTab'
    store_quantity_candidate(TS_name)
    store_quanitity_column_type_result(TS_name)
