import pickle
import ast
from date_identification import is_year_mention,is_year,is_date_mention
import numpy as np
from candidate_generate import Candidate,read_index_info,quantity_candidate_type,str_to_bool,jaccard_similarity#,traverse_literal_to_entity
import os
from unlinkable_sample import read_new_tableset
import json
from quantity_candidate_generate import quantity_mention_candidate
from quantity_column_type import quantity_column_type
def QuantityType(column_type_name,col,bert_v,ic,core,h,header_s,qcol,total_s):

    Q={}
    Q['name']=column_type_name
    Q['col']=col
    Q['bv']=bert_v
    Q['ic']=ic
    Q['core']=core
    Q['h']=h
    Q['hs']=header_s
    Q['qcol']=qcol
    Q['ts']=total_s
    
    return Q


def entity_type_filter(WT,ctf,NCS):
    
    for c in range(WT['col_num']):
        
        if str(c) not in WT['qcol']:
            
            f=[]
            #print(ctf.keys())
            if str(c) not in ctf:
                ctf[str(c)]=[]
            for t in ctf[str(c)]:
                
                flag=0
                for pos in NCS:
                    
                    if int(pos.split('_')[1])==c:
                        
                        for can in NCS[pos]:
                            
                            if t in can['type'] or t in can['rin']:
                                
                                flag=1
                f.append(flag)
            if 1 in f:
                ctf[str(c)]=[ctf[str(c)][x] for x in range(len(f)) if f[x]==1]
            
    return ctf
                           
        
def quantity_type_filter(WT,QCT,Col,QT):
    
    QC={}
    ec=[i[0] for i in Col]
    qc=[i[1] for i in Col]
    ecol=[]
    for c in WT['qcol']:
        QC[int(c)]=[]
        if WT['qflag'][int(c)]==1:
            
            if int(c) in qc:
                
                ecol=ec[qc.index(int(c))]
                for t in QCT[ecol]:
                    
                    if QT[t]==False:
                        
                        QC[int(c)].append(t)
            else:
                
                QC[int(c)]=[t for t in QCT[int(WT['core'])] if QT[t]==False]
            
                        
        elif WT['qflag'][int(c)]==-1:
            
            if int(c) in qc:
                
                ecol=ec[qc.index(int(c))]
                for t in QCT[ecol]:
                    
                    if QT[t]==True:
                        
                        QC[int(c)].append(t)
            else:
                
                QC[int(c)]=[t for t in QCT[int(WT['core'])] if QT[t]==True]
                
                        
    return QC

def topk_bert_column_type_score(QCB,TB,c):
    
    S_w=np.zeros(len(TB[0]))
    S_s=np.zeros(len(TB[0]))
    
    for i in range(len(QCB['whole_row'])):
        
        TS=np.array(QCB['whole_row'][i])
        Big_TS=np.tile(TS,(len(TB[0]),1))
        #print(np.shape(Big_TS),np.shape(d[1]))
        Diff=(Big_TS-TB[1])**2
        #print(np.shape(Diff))
        S_w+=np.sum(Diff,axis=1)
    #print(S_w,'sw')
    for i in range(len(QCB['single_word'])):
        
        TS=np.array(QCB['single_word'][i][c])
        
        #print(TS[0][0],i)
        Big_TS=np.tile(TS,(len(TB[0]),1))
        
        #print(np.shape(Big_TS),np.shape(TB[1]))
        if np.shape(Big_TS)[0]==1:
            Big_TS=Big_TS[0]
            #print('y')
        Diff=(Big_TS-TB[1])**2
        
        #print(np.shape(Diff),np.shape(np.sum(Diff,axis=1)),np.shape(S_s),i)
        S_s+=np.sum(Diff,axis=1)
    #print(S_s,'ss')    
    S=S_w+S_s
    LI=np.argsort(S)
    TI=LI[:10]
    SI=np.zeros(10)
    #print(TI)
    Q=[]
    for i in range(10):
        
        j=TI[i]
        Q.append(TB[0][j])
        SI[i]=S[j]
        
    return Q,SI

def topk_bert_header_score(QCB,H,c):
    
    S_w=[]
    
    for i in range(len(QCB['whole_row'])):
        
        TS=np.array(QCB['whole_row'][i])
        #Big_TS=np.tile(TS,(len(TB[0]),1))
        #print(np.shape(Big_TS),np.shape(d[1]))
        Diff=(np.array(H)-TS)**2
        #print(np.shape(Diff))
        #S_w+=np.sum(Diff)
        S_w.append(np.sum(Diff))
    
    for i in range(len(QCB['single_word'])):
        
        TS=np.array(QCB['single_word'][i][c])
        #Big_TS=np.tile(TS,(len(TB[0]),1))
        #print(np.shape(Big_TS),np.shape(d[1]))
        #print(np.shape(H),np.shape(TS))
        if np.shape(TS)[0]==1:
            TS=TS[0]
            #print('y')
            #print(np.shape(TS))
        #print(np.shape(H),np.shape(TS))

        Diff=(np.array(H)-TS)**2
        #print(np.shape(Diff))
        #print(np.shape(TS))
        #print(np.shape(S_s),i)
        #S_s.append(np.sum(Diff))
        #print(np.shape(Diff))
        #S_s+=np.sum(Diff)
        S_w[i]+=np.sum(Diff)
    
    #S=S_w+S_s
        
    return S_w

def total_quantity_score(QCB,H,c):
    
    S=[]
        
    for i in range(len(QCB['whole_row'])):
        
        TS=np.array(QCB['whole_row'][i])
        #Big_TS=np.tile(TS,(len(TB[0]),1))
        #print(np.shape(Big_TS),np.shape(d[1]))
        Diff=(np.array(H)-TS)**2
        #print(np.shape(Diff))
        S.append(np.mean(Diff))
    #print(S,'SW')
    #D=0
    for i in range(len(QCB['single_word'])):
        
        TS=np.array(QCB['single_word'][i][c])
        if np.shape(TS)==(1,1,768):
            TS=TS[0]
        #Big_TS=np.tile(TS,(len(TB[0]),1))
        #print(np.shape(TS),np.shape(H))
        Diff=(np.array(H)-TS)**2
        if np.isnan(np.mean(Diff)):
            #print(S[i-1],'i-1')
            S[i]+=0
        else:
            #print(np.mean(Diff,axis=1)[0]!=np.nan,np.mean(Diff,axis=1)[0])
            S[i]+=np.mean(Diff)
    #print(S,'SS')            
    return S
def read_new_QCT(QCT):
    
    qct={}
    for col in QCT:
        qct[int(col)]=QCT[col]
        
    return qct
    
def initial_type_node(WT,NCS,QT,QB,ic,db_name,yd,sd,ctf,QCB,QCT,Col,HB,arg1):
    
    ctf=entity_type_filter(WT,ctf,NCS)
    QC=quantity_type_filter(WT,QCT,Col,QT)
    #Col=set(Col)
    Col_q=[i[1] for i in Col]
    Col_e=[i[0] for i in Col]
    #print(type())
    ctfq={}
    for c in WT['qcol']:
        
        if int(c) in Col_q:
            
            cc=Col_e[Col_q.index(int(c))]
            
        else:
            cc=int(WT['core'])
        
        ctfq[int(c)]=[]
        if len(QC[int(c)])!=0:
            
            for t in QC[int(c)]:
                
                if WT['qflag'][int(c)]==1:
                    qcol='scale'
                elif WT['qflag'][int(c)]==-1:
                    M=[i[int(c)] for i in WT['con'] if int(c)<len(i)]
                    if is_year(M,is_year_mention,0.8):
                        qcol='year'
                        
                        if 'year' not in t and 'Year' not in t:
                            #print(t)
                            continue
                    elif is_year(M,is_date_mention,0.8):
                        qcol='date'
                        if 'date' not in t and 'Date' not in t:
                            continue
                    else:
                        qcol=''
                #print(qcol)
                #s=np.mean(np.array(total_quantity_score(QCB,QB[1][QB[0].index(t)],c)))
                s=total_quantity_score(QCB,QB[1][QB[0].index(t)],c)
                #print(s)
                header=WT['header']
                ct=QuantityType(t,int(c),QB[1][QB[0].index(t)].tolist(),ic[t],cc,header[int(c)],0,qcol,s)
                #print(s)
                header_s=0.25*np.mean((QB[1][QB[0].index(t)]-np.array(HB[c]))**2)+0.75*(1-jaccard_similarity(t,header[int(c)]))
                ct['hs']=header_s
                #print(header_s)
                #print(ct.qcol)
                ctfq[int(c)].append(ct)
                
                        
        else:
            
            H=WT['header']
            
            if WT['qflag'][int(c)]==1:
                
                Q,_=topk_bert_column_type_score(QCB,sd,c)
                for i in range(len(Q)):
                    t=Q[i]
                    b=sd[1][sd[0].index(t)]
                    #print(HB.keys())
                    header_s=0.25*np.mean((QB[1][QB[0].index(t)]-np.array(HB[str(c)]))**2)+0.75*(1-jaccard_similarity(t,H[int(c)]))
                    TI=total_quantity_score(QCB,QB[1][QB[0].index(t)],c)
                    ct=QuantityType(t,int(c),b.tolist(),ic[t],int(WT['core']),H[int(c)],header_s,'scale',TI)
                    ctfq[int(c)].append(ct)
                s=topk_bert_header_score(QCB, HB[str(c)],c)
                #print(s)
                h=QuantityType(H[int(c)],int(c),HB[str(c)],-1,cc,H[int(c)],0,'scale',s)
                ctfq[int(c)].append(h)
                #print(ctfq[int(c)][-1]['ts'])
            else:
                Q,_=topk_bert_column_type_score(QCB,yd,c)
                #print(Q,TI)
                for i in range(len(Q)):
                    t=Q[i]
                    #print(t)
                    b=yd[1][yd[0].index(t)]
                    M=[i[int(c)] for i in WT['con']]
                    if is_year(M,is_year_mention,0.8):
                        qcol='year'
                        if 'year' not in t and 'Year' not in t:
                            continue
                    elif is_year(M,is_date_mention,0.8):
                        qcol='date'
                        if 'date' not in t and 'Date' not in t:
                            continue
                    else:
                        qcol=''
                    header_s=0.25*np.mean((QB[1][QB[0].index(t)]-np.array(HB[str(c)]))**2)+0.75*(1-jaccard_similarity(t,H[int(c)]))
                    TI=total_quantity_score(QCB,QB[1][QB[0].index(t)],c)
                    ct=QuantityType(t,int(c),b.tolist(),ic[t],cc,H[int(c)],header_s,qcol,TI)
                    ctfq[int(c)].append(ct)
                    #print(TI[i],'w')
                s=topk_bert_header_score(QCB, HB[str(c)],c)
                #print(s[0],'s')
                h=QuantityType(H[int(c)],int(c),HB[str(c)],-1,cc,H[int(c)],0,qcol,s)
                #print(s)
                ctfq[int(c)].append(h)
        
        if ctfq[int(c)]==[]:
            
            H=WT['header']
            
            if WT['qflag'][int(c)]==1:
                
                Q,_=topk_bert_column_type_score(QCB,sd,c)
                for i in range(len(Q)):
                    t=Q[i]
                    b=sd[1][sd[0].index(t)]
                    header_s=0.25*np.mean((QB[1][QB[0].index(t)]-np.array(HB[str(c)]))**2)+0.75*(1-jaccard_similarity(t,H[int(c)]))
                    TI=total_quantity_score(QCB,QB[1][QB[0].index(t)],c)
                    ct=QuantityType(t,int(c),b.tolist(),ic[t],int(WT['core']),H[int(c)],header_s,'scale',TI)
                    ctfq[int(c)].append(ct)
                s=topk_bert_header_score(QCB, HB[str(c)],c)
                h=QuantityType(H[int(c)],int(c),HB[str(c)],-1,cc,H[int(c)],0,'scale',s)
                ctfq[int(c)].append(h)
            else:
                
                Q,_=topk_bert_column_type_score(QCB,yd,c)
                #print(Q,TI)
                for i in range(len(Q)):
                    t=Q[i]
                    #print(t)
                    b=yd[1][yd[0].index(t)]
                    M=[i[int(c)] for i in WT['con']]
                    if is_year(M,is_year_mention,0.8):
                        qcol='year'
                        if 'year' not in t and 'Year' not in t:
                            continue
                    elif is_year(M,is_date_mention,0.8):
                        qcol='date'
                        if 'date' not in t and 'Date' not in t:
                            continue
                    else:
                        qcol=''
                    header_s=0.25*np.mean((QB[1][QB[0].index(t)]-np.array(HB[str(c)]))**2)+0.75*(1-jaccard_similarity(t,H[int(c)]))
                    TI=total_quantity_score(QCB,QB[1][QB[0].index(t)],c)
                    ct=QuantityType(t,int(c),b.tolist(),ic[t],cc,H[int(c)],header_s,qcol,TI)
                    #print(TI[i])
                    ctfq[int(c)].append(ct)
                s=topk_bert_header_score(QCB, HB[str(c)],c)
                #print(s[0])
                h=QuantityType(H[int(c)],int(c),HB[str(c)],-1,cc,H[int(c)],0,qcol,s)
                ctfq[int(c)].append(h)       
                
    return ctf,ctfq,Col_e,Col_q
            
                        
def read_type_information():
    
    f=open('../KG/KG_data/type_quantity_classified.data','rb')
    QT=pickle.load(f)
    f.close()
    
    f=open('type_bert_vector.data','rb')
    QB=pickle.load(f)
    f.close()
    
    f=open('../KG/KG_data/ic_type.data','rb')
    ic=pickle.load(f)
    f.close()
    
    f=open('../KG/KG_data/scale_bert.data','rb')
    sd=pickle.load(f)
    f.close()
    
    f=open('../KG/KG_data/year_bert.data','rb')
    yd=pickle.load(f)
    f.close()
    
    return QT,QB,ic,yd,sd
    

def add_quantity_candidate(NCS,ctfq,yd,WT,td,wd):
    
    for c in ctfq:
        
        if ctfq[c][0]['qcol']=='year':
            
            for r in range(len(WT['con'])):
                #print((r,c))
                if str(r)+'_'+str(c) not in NCS:
                    
                    m=WT['con'][r][c]
                    #print(m)
                    can=Candidate(str(m),str(m),[],(r,c),0,False,[],[])
                    can=quantity_candidate_type(can,td,wd)
                    NCS[str(r)+'_'+str(c)]=[can]
                elif NCS[str(r)+'_'+str(c)]!=[]:
                    
                    for k in range(len(NCS[str(r)+'_'+str(c)])):
                        
                        #can=NCS[str(r)+'_'+str(c)][k]
                        can=Candidate(NCS[str(r)+'_'+str(c)][k],NCS[str(r)+'_'+str(c)][k],[],(r,c),0,False,[],[])
                        can=quantity_candidate_type(can,td,wd)
                        NCS[str(r)+'_'+str(c)][k]=can
                else:
                    m=int(WT['con'][r][c])
                    can=Candidate(str(m),str(m),[],(r,c),0,False,[],[])
                    can=quantity_candidate_type(can,td,wd)
                    NCS[str(r)+'_'+str(c)].append(can)
                
                    
    return NCS

def read_candidate_filter(WT,fil_can_dir):
    
    #fil_can_dir='处理后/处理后/graph_candidate/'
    f=open(fil_can_dir+WT['id']+'_NCS.txt','r',encoding='utf-8')
    NCS={}
    for l in f.readlines():
        l=l.strip('\n')
        #print(l)
        S=l.split('\t')
        if S[0]=='pos':
            pos=ast.literal_eval(S[1])
            NCS[pos]=[]
            loc=-1
        elif S[0]=='c_type':
            NCS[pos][loc].c_type=[s for s in S[1:]]
            NCS[pos][loc].c_type=list(filter(lambda x: x != '', NCS[pos][loc].c_type))
        elif S[0]=='c_rin':
            NCS[pos][loc].rin=[s for s in S[1:]]
            NCS[pos][loc].rin=list(filter(lambda x: x != '', NCS[pos][loc].rin))
        elif S[0]=='c_rq':
            for s in S[1:]:
                if s!='':
                    s=s.strip('(')
                    s=s.strip(')')
                    sl=s.split(',')
                    sl0=sl[0].replace("'","")
                    sl1=sl[1].replace("'","")
                    sl1=sl1.strip(' ')
                    tur=(sl0,sl1)
                    NCS[pos][loc].rq.append(tur)
            #NCS[pos][loc].rq=list(filter(lambda x: x != '', NCS[pos][loc].rq))
        elif S[0]=='qcan':
            can=Candidate(S[1],WT['con'][pos[0]][pos[1]],[],pos,0,True,[],[])
            loc=len(NCS[pos])
            NCS[pos].append(can)
        else:
            can=Candidate(S[0],S[1],[],pos,float(S[2]),str_to_bool(S[3]),[],[])
            loc=len(NCS[pos])
            NCS[pos].append(can)
    return NCS

def store_quantity_candidate(TS_name,arg1):
    
    if arg1==None:
        table_dir='Data/'
        can_dir='Candidate/'
        col_type_dir='ColumnType/'
        bert_dir='BERT/'
    else:
        if 'null' in arg1:
            #print(args)
            r=arg1.split('_')[-1]
            table_dir='处理后/处理后/'+'null_table/Rate_'+r+'/Data/'
            can_dir='处理后/处理后/'+'null_table/Rate_'+r+'/Candidate/'
            col_type_dir='处理后/处理后/'+'null_table/Rate_'+r+'/ColumnType/'
            bert_dir='处理后/处理后/'+'null_table/Rate_'+r+'/BERT/'
        else:
            #print(args)
            r=arg1.split('_')[-1]
            table_dir='处理后/处理后/'+'sample_table/Num_'+r+'/Data/'
            can_dir='处理后/处理后/'+'sample_table/Num_'+r+'/Candidate/'
            col_type_dir='处理后/处理后/'+'sample_table/Num_'+r+'/ColumnType/'
            bert_dir='处理后/处理后/'+'sample_table/Num_'+r+'/BERT/'
            
    with open(table_dir+TS_name+'_table.json','r',encoding='utf-8') as file:
        TableSet=json.load(file)
    with open(can_dir+TS_name+'_fcs.json','r',encoding='utf-8') as file:
        Can_set=json.load(file)
    with open(col_type_dir+TS_name+'_ctf.json','r',encoding='utf-8') as file:
        CT=json.load(file)
        
    
    with open(bert_dir+TS_name+'_qcb.json','r',encoding='utf-8') as file:
        QCB=json.load(file)
    with open('BERT/'+TS_name+'_hb.json','r',encoding='utf-8') as file:
        HB=json.load(file)
    #QCB,HB=read_column_type_whole(TS_name)
    QT,QB,ic,yd,sd=read_type_information()
    #td,wd=read_index_info()
    
    
    try:
        with open(can_dir+TS_name+'_pcan.json','r',encoding='utf-8') as file:
            C=json.load(file)
        with open(col_type_dir+TS_name+'_qct.json','r',encoding='utf-8') as file:
            QCT=json.load(file)
        with open(can_dir+TS_name+'_col.json','r',encoding='utf-8') as file:
            Col=json.load(file)
    except:
        #PC={}
        C={}
        QCT={}
        Col={}
    j=0
    for i in TableSet:
        #print(i)
        if i in C and C[i]!='Fail':
            continue
        if 1>0:
            j+=1
            W=TableSet[i]
            if 'core' not in W:
                W['core']=0
            QC=quantity_mention_candidate(W)
            NCS,qct,col=quantity_column_type(W,CT[i],Can_set[i],QC)
            #print(W['core'])
            ctf,ctfq,_,_=initial_type_node(W,NCS,QT,QB,ic,TS_name,yd,sd,CT[i],QCB[i],qct,col,HB[i],arg1)
            td,wd=read_index_info()
            ncs=add_quantity_candidate(NCS,ctfq,yd,W,td,wd)
            C[i]=ncs

            QCT[i]=qct
            Col[i]=col
        else:
            C[i]='Fail'
            QCT[i]='Fail'
            Col[i]='Fail'
            print(i,'fail')
        if j%100==0:
            with open(can_dir+TS_name+'_pcan.json','w') as f:
                json.dump(C,f)
            with open(col_type_dir+TS_name+'_qct.json','w') as f:
                json.dump(QCT,f)
            with open(can_dir+TS_name+'_col.json','w') as f:
                json.dump(Col,f)
    with open(can_dir+TS_name+'_pcan.json','w') as f:
        json.dump(C,f)
    with open(col_type_dir+TS_name+'_qct.json','w') as f:
        json.dump(QCT,f)
    with open(can_dir+TS_name+'_col.json','w') as f:
        json.dump(Col,f)


if __name__=="__main__":
    for r in ['6','8']:
        store_quantity_candidate('T2DC','sample_'+r)

    