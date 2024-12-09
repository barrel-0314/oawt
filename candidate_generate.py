import pickle
#from class_candidate import Candidate
from index_tree import find_triple,read_tree
import os
from string_index_test import candidate_index
from utils import mention_pre
#import re
import ast
import json
import copy
from transformers import BertModel,BertTokenizer
from bert_score import mention_can_diff


def Candidate(name,mention,t,pos,bert_s,lit,rin,rq):
    
    C={}
    C['name']=name
    C['mention']=mention
    C['type']=t
    C['pos']=pos
    C['bs']=bert_s
    C['lit']=lit
    C['rin']=rin
    C['rq']=rq
    
    return C
 
def other_dir(label):
    
    if len(label)==0 or ord(label[0])>1000:        
        dire='other_1000'
    elif '1'<=label[0]<='9':
        if label[0]!='1'and label:
            dire='other_'+label[0]
        else:
            if len(label)==1:
                dire='other_non'
            elif '0'<=label[1]<='8':
                dire='other_08'
            elif label[1]=='9':
                dire='other_91'
            else:
                dire='other_non'
    elif label[0]=='0':
        
        dire='other_non'
    elif ord(label[0])<=100:
        dire='other_100'
    else:
        dire='other_500'
    
    return dire


def jaccard_similarity(str1, str2):
    set1 = set(str1.upper())
    set2 = set(str2.upper())
    
    # 计算两个集合的并集和交集大小
    union_size = len(set1 | set2)
    intersection_size = len(set1 & set2)
    
    # 计算Jaccard相似度
    if union_size == 0:
        return 0.0
    else:
        return intersection_size / union_size        
        

    
    
def find_distance_inlist(L,label,n,c,lit):
    
    if lit==False:
        for l in L:
        
            firstsplit=l.split('/')
            entity1=firstsplit[-1]
            #entity2=(firstsplit[-1].split('>'))[0]
        
            if entity1 not in c.keys():
            
                if jaccard_similarity(label,entity1)>n:
                
                    c[entity1]=jaccard_similarity(label,entity1)
                    
    else:
        for l in L:
            
            if l not in c.keys():
                #print(jaccard_similarity(label,l))
                if jaccard_similarity(label,l)>n:
                    c[l]=jaccard_similarity(label,l)

    return c

def candidate_search_in_literal(label,k,n):
    result=[]
    candidate={}
    
    s_index=candidate_index(label,True)
    #print(s_index,label)
    for ff in s_index:
        
        if len(ff)==0:
            S=other_dir(ff)
        else:
            
            label2=ff[0]
            s=label2[0].upper() 
            if s>='A' and s<='Z':
                S=s
            else:
                S=other_dir(ff)
                
        flen=len(s_index[ff])
        for l in range(flen):
            
            st=s_index[ff][l][0]
            ed=s_index[ff][l][1]
            f=open('../KG/list_data_char_index/'+S+'_ll/'+str(st)+'_to_'+str(ed)+'.data','rb')
            ll=pickle.load(f)
            f.close()
            candidate=find_distance_inlist(ll,label,n,candidate,True)
        
    #candidate=find_distance_inlist(L,label,n,candidate,True)
    candidate=sorted(candidate.items(),key=lambda kv:(kv[1],kv[0]),reverse=True)
    #print(candidate)
    if len(candidate)>k:
        
        candidate=candidate[0:k]
    result=[c[0] for c in candidate]       
    return result    



def identify_file(candidate):

    if len(candidate)==0:
        ch='other'
    else:
        if 'A'<=candidate[0].upper()<='Z':
            ch=candidate[0].upper()
        else:
            ch='other'
            
    return ch

def candidate_search_simplify(label,s_index,k,n):
    
    result=[]
    candidate={}
    #n=len(label)
    
    #flen=len(s_index)
    for ff in s_index:
        
        if len(ff)==0:
            S='other'
        else:
            
            label2=ff[0]
            s=label2[0].upper() 
            if s>='A' and s<='Z':
                S=s
            else:
                S='other'
                
        flen=len(s_index[ff])
        for l in range(flen):
            st=s_index[ff][l][0]
            ed=s_index[ff][l][1]
            f=open('../KG/list_data_char_index/'+S+'_el/'+str(st)+'_to_'+str(ed)+'.data','rb')
            el=pickle.load(f)
            f.close()
            can=find_distance_inlist(el,label,n,candidate,False)
            candidate.update(can)
    
    candidate=sorted(candidate.items(),key=lambda kv:(kv[1],kv[0]),reverse=True)

    if len(candidate)>k:
        
        candidate=candidate[0:k]
        
    result=[c[0] for c in candidate]
    return result


          
def candidate_type(Can_set,tree_dict,word_dict):
    
    indexfile='../KG/index KG/'
    for pos in Can_set:
        
        for i in range(len(Can_set[pos])):
            
            c=Can_set[pos][i]
            if c['lit']==True:
                continue
            if len(c['name'])==0:
                cc='o'
            elif c['name'][0]<='Z' and c['name'][0]>='A':
                cc='az'
            else:
                cc='o'
            
            Can_set[pos][i]['type']=list(set(find_triple(c['name'],True,tree_dict['ty_'+cc],word_dict['ty_'+cc],'ty','o',indexfile)))
            Can_set[pos][i]['rin']=list(set(find_triple(c['name'],True,tree_dict['tr_'+cc],word_dict['tr_'+cc],'tr','p',indexfile)))
            Can_set[pos][i]['rq']=list(set(find_triple(c['name'],True,tree_dict['ti_'+cc],word_dict['ti_'+cc],'ti','po',indexfile)))
            Can_set[pos][i]['rq']+=list(set(find_triple(c['name'],True,tree_dict['li_'+cc],word_dict['li_'+cc],'li','po',indexfile)))
    return Can_set

def quantity_candidate_type(c,tree_dict,word_dict):
    
    indexfile='../KG/index KG/'
    if len(c['name'])==0:
        cc='o'
    elif c['name'][0]<='Z' and c['name'][0]>='A':
        cc='az'
    else:
        cc='o'
    
    c['type']=list(set(find_triple(c['name'],True,tree_dict['ty_'+cc],word_dict['ty_'+cc],'ty','o',indexfile)))
    c['rin']=list(set(find_triple(c['name'],True,tree_dict['tr_'+cc],word_dict['tr_'+cc],'tr','p',indexfile)))
    c['rin']+=list(set(find_triple(c['name'],False,tree_dict['lr_'+cc],word_dict['lr_'+cc],'lr','p',indexfile)))
    c['rin']=list(set(c['rin']))
    for k in range(len(c['type'])):
        
        c['type'][k]=c['type'][k].split('/')[-1]
    for k in range(len(c['rin'])):
        
        c['rin'][k]=c['rin'][k].split('/')[-1]
    
    return c

def candidate_literal_property(Can_set,tree_dict,word_dict):
    indexfile='../KG/index KG/'
    for pos in Can_set:
        
        for i in range(len(Can_set[pos])):
            
            c=Can_set[pos][i]
            if c['lit']==False:
                continue
            if len(c['name'])==0:
                cc='o'
            elif c['name'][0]<='Z' and c['name'][0]>='A':
                cc='az'
            else:
                cc='o'
            
            Can_set[pos][i]['rin']=list(set(find_triple(c['name'],False,tree_dict['lr_'+cc],word_dict['lr_'+cc],'lr','p',indexfile)))            
    return Can_set

def read_index_info():
    word_dict={}
    tree_dict={}
    fl=['ti','tr','li','lr','ty']
    wl=['az','o']
    
    for ff in fl:
        
        for ww in wl:
            
            T,L=read_tree(ff+'_wl_'+ww)
            word_dict[ff+'_'+ww]=L
            tree_dict[ff+'_'+ww]=T
                     
    return tree_dict,word_dict
         

def initial_candidate(WT,C,model,tokenizer):

    Can_set={}
    for r in range(len(WT['con'])):
        for c in range(WT['col_num']):
            
            pos=(r,c)
            if c>=len(WT['con'][r]):
                mention=''
            else:
                mention=WT['con'][r][c]
            mention=mention.strip(' ')
            M=mention_pre(mention)
            Can_set[str(r)+'_'+str(c)]=[]
                
            for men in M:
                if men in C['entity']:
                    for i in range(len(C['entity'][men])):
                        if ' ' in C['entity'][men][i]:
                            s=C['entity'][men][i].replace(' ','_')
                        else:
                            s=C['entity'][men][i]
                        bs=mention_can_diff(men, s, model, tokenizer)
                        norm_c=Candidate(s,WT['con'][pos[0]][pos[1]],[],pos,bs,False,[],[])
                        Can_set[str(r)+'_'+str(c)].append(norm_c)
                if men in C['literal']:
                    for i in range(len(C['literal'][men])):
                        if '_' in C['literal'][men][i]:
                            s=C['literal'][men][i].replace('_',' ')
                        else:
                            s=C['literal'][men][i]
                        bs=mention_can_diff(men, s, model, tokenizer)
                        norm_c=Candidate(s,WT['con'][pos[0]][pos[1]],[],pos,bs,True,[],[])
                        Can_set[str(r)+'_'+str(c)].append(norm_c)

    tree_dict,word_dict=read_index_info()
    #add candidate type    
    Can_set=candidate_type(Can_set,tree_dict,word_dict)
    Can_set=candidate_literal_property(Can_set,tree_dict,word_dict)
    
    Can_set=candidate_filter(Can_set)
     
    return Can_set,tree_dict,word_dict

def candidate_filter_equal(NCS):

    for pos in NCS:
        
        flag=0
        for can in NCS[pos]:
            
            if can['name']==can['mention'] or can['name'].upper()==can['mention'].upper():
                
                flag=1
                break
            
            elif can['name'].replace('_',' ')==can['mention'] or can['name'].replace('_',' ').upper()==can['mention'].upper():
                flag=1
                break
        if flag==1:
            L=[]
            for can in NCS[pos]:
                if can['name']==can['mention'] or can['name'].upper()==can['mention'].upper():
                    
                    L.append(can)
                    
                elif can['name'].replace('_',' ')==can['mention'] or can['name'].replace('_',' ').upper()==can['mention'].upper():
                    L.append(can)
                    
                
            NCS[pos]=L
            
    return NCS

def candidate_filter_equal_literal(NCS):
    
    for pos in NCS:
        
        flag=0
        for can in NCS[pos]:
            
            if can['name'].replace('_',' ')==can['mention'].replace('_',' ') and can['lit']==False:
                flag=1
                break
            elif can['name'].replace('_',' ').upper()==can['mention'].replace('_',' ').upper() and can['lit']==False:
                flag=1
                break
        if flag==1:
            #print(len(NCS[pos]))
            L=[]
            for can in NCS[pos]:
                
                if can['name']==can['mention'] and can['rin']==['name']:
                    
                    if can['lit']==True:
                        
                        continue
                    else:
                        L.append(can)
                elif can['name'].replace(' ','_').upper()==can['mention'].replace(' ','_').upper() and can['rin']==['name']:
                    
                    if can['lit']==True:
                        
                        continue
                    else:
                        L.append(can)  
                else:
                    L.append(can)
                '''
                if can['name']==can['mention'] and can['rin']!=['name']:
                    
                    L.append(can)
                elif can['name'].upper()==can['mention'].upper() and can['rin']!=['name']:
                    
                    L.append(can)
                elif can['name'].replace('_',' ')==can['mention']:
                    L.append(can)
                elif can['name'].replace('_',' ').upper()==can['mention'].upper():
                    L.append(can)
                '''
            NCS[pos]=L
    return NCS

def traverse_literal_to_entity(NCS,WT,td,wd):
    
    indexfile='../KG/index KG/'
    for pos in NCS:
        
        can_entity=[can['name'] for can in NCS[pos] if can['lit']==False]
        delete_can1=[]
        delete_can2=[]
        add_can=[]
        for can in NCS[pos]:
            
            if can['name'].replace(' ','_') in can_entity and can['lit']==True:
                
                delete_can1.append(can)
                
            elif can['lit']==True and can['name'].replace(' ','_') not in can_entity:
                if 'name' in can['rin']:
                    
                    if len(can['name'])==0:
                        cc='o'
                    elif can['name'][0].upper()>='A' and can['name'][0].upper()<='Z':
                        cc='az'
                    else:
                        cc='o'
                    pred=list(find_triple(can['name'],True,td['lr_'+cc],wd['lr_'+cc],'lr','p',indexfile))
                    enti=list(find_triple(can['name'],True,td['lr_'+cc],wd['lr_'+cc],'lr','s',indexfile))
                    entity=''
                    for k in range(len(pred)):
                        str_pred=str(pred[k]).split('/')[-1]
                        if str_pred=='name':
                            
                            entity=str(enti[k]).split('/')[-1]
                            break
                    if entity!='':
                        delete_can2.append(can)
                        #print(can['name'])
                        
                        can_e=Candidate(entity,can['mention'],[],(int(pos.split('_')[0]),int(pos.split('_')[1])),can['bs'],False,[],[])
                        #print(can_e['name'])
                        can_e['type']=list(set(find_triple(can_e['name'],True,td['ty_'+cc],wd['ty_'+cc],'ty','o',indexfile)))
                        can_e['rin']=list(set(find_triple(can_e['name'],True,td['tr_'+cc],wd['tr_'+cc],'tr','p',indexfile)))
                        can_e['rq']=list(set(find_triple(can_e['name'],True,td['ti_'+cc],wd['ti_'+cc],'ti','po',indexfile)))
                        can_e['rq']+=list(set(find_triple(can_e['name'],True,td['li_'+cc],wd['li_'+cc],'li','po',indexfile)))
                        add_can.append(can_e)
                elif can['rin']==[]:
                    
                    if len(can['name'])==0:
                        cc='o'
                    elif can['name'][0].upper()>='A' and can['name'][0].upper()<='Z':
                        cc='az'
                    else:
                        cc='o'
                    #typ=list(find_triple(can['name'],True,td['ty_'+cc],wd['ty_'+cc],'ty','o',indexfile))
                    enti=list(find_triple(can['name'].replace(' ','_'),True,td['ty_'+cc],wd['ty_'+cc],'ty','o',indexfile))
                    entity=''
                    if len(enti)!=0:
                        entity=str(can['name'].replace(' ','_')).split('/')[-1]
                    if entity!='':
                        delete_can2.append(can)
                        #print(can['name'])
                        can_e=Candidate(entity,can['mention'],[],(int(pos.split('_')[0]),int(pos.split('_')[1])),can['bs'],False,[],[])
                        #print(can_e['name'])
                        can_e['type']=list(set(find_triple(can_e['name'],True,td['ty_'+cc],wd['ty_'+cc],'ty','o',indexfile)))
                        can_e['rin']=list(set(find_triple(can_e['name'],True,td['tr_'+cc],wd['tr_'+cc],'tr','p',indexfile)))
                        can_e['rq']=list(set(find_triple(can_e['name'],True,td['ti_'+cc],wd['ti_'+cc],'ti','po',indexfile)))
                        can_e['rq']+=list(set(find_triple(can_e['name'],True,td['li_'+cc],wd['li_'+cc],'li','po',indexfile)))
                        add_can.append(can_e)
        #print(delete_can1,delete_can2,add_can)  
        if len(delete_can1)!=0:
            for j in range(len(delete_can1)):
                d_can=delete_can1[j]
                NCS[pos].remove(d_can)
        if len(delete_can2)!=0:
            
            for j in range(len(delete_can2)):
                d_can=delete_can2[j]
                NCS[pos].remove(d_can)
                NCS[pos].append(add_can[j])
                
    return NCS

def preprocess_candidate_type(NCS,QT):
    
    for pos in NCS:
        for k in range(len(NCS[pos])):
            can=NCS[pos][k]
            c_type=[]
            c_rin=[]
            c_rq=[]
            for t in can['type']:
                t_s=str(t).split('/')[-1]
                c_type.append(t_s)
                
            for t in can['rin']:
                t_s=str(t).split('/')[-1]
                c_rin.append(t_s)
                
            for t in can['rq']:
                t_s=str(t[0]).split('/')[-1]
                o_s=str(t[1]).split('/')[-1]
                if t_s in QT:
                    c_rq.append((t_s,o_s))
            can['type']=c_type
            can['rin']=c_rin
            can['rq']=c_rq
            NCS[pos][k]=can
    return NCS

def candidate_main(WT,n):
    C={}
    for r in range(len(WT['con'])):
        #print(r,WT.column_num)
        for c in range(WT['col_num']):
            if len(WT['con'][r])<=c:
                continue

            mention=WT['con'][r][c]
            mention=mention.strip(' ')
            M=mention_pre(mention)
            #print(M)
            for men in M:
                if men not in C:
                    
                    if len(men)!=0:
                        s_index=candidate_index(men, False)
                        re=candidate_search_simplify(men,s_index,5,n)
                    else:
                        re=[]
                    C[men]=re
    
    return C

def candidate_main_literal(WT,n):
    
    C={}
    for r in range(len(WT['con'])):
        #print(r,WT.column_num)
        for c in range(WT['col_num']):
            #print(len(WT['con'][r]),c)
            if len(WT['con'][r])<=c:
                continue
            mention=WT['con'][r][c]
            mention=mention.strip(' ')
            M=mention_pre(mention)
            for men in M:
                if men not in C:
                    
                    if len(men)!=0:
                        res=candidate_search_in_literal(men,5,n)
                    else:
                        res=[]
                    C[men]=res   
    return C

    
def store_candidate_set_literal(TableSet_name,n,recommend):
    
    
    with open('Data/'+TableSet_name+'_table.json','r',encoding='utf-8') as file:
        TableSet=json.load(file)
    #Can={}
    j=0
    try:
        with open('Candidate/'+TableSet_name+'_table_cpl.json','r',encoding='utf-8') as file:
            Can=json.load(file)
    except:
        Can={}
    if recommend==False:
        for i in TableSet:
            j+=1
            #if j>10:
            #    break
            if Can[i]!='Fail':
                continue
            print(i,j)
            if 1>0:
                #CS={}
                W=TableSet[i]
                #print(W['con'])
                #M=TableSet.table_name[i]
                Can[i]={}
                Rl=candidate_main_literal(W,n)
                Can[i]['literal']=Rl
                Can[i]['entity']=candidate_main(W,n)
            else:
                Can[i]='Fail'
                print('fail')
        with open('Candidate/'+TableSet_name+'_table_cpl2.json','w') as f:
            json.dump(Can,f)
    return Can
def candidate_filter(Can_set):
    
    Can_set_copy={}
    for pos in Can_set:
        Can_set_copy[pos]=[]
        L=Can_set[pos]
        flag=False
        Loc=[]
        for i in range(len(L)):
            c=L[i]
            s1=c['name'].replace('_',' ')
            s1=s1.upper()
            s2=c['mention'].replace('_',' ')
            s2=s2.upper()
            if jaccard_similarity(s1,s2)==1:
                
                flag=True
                Loc.append(i)
                
            elif c['bs']==0:
                
                flag=True
                Loc.append(i)
                
        if flag==True:
            
            L_copy=[]
            for i in range(len(L)):
                
                if i in Loc:
                    
                    L_copy.append(L[i])
                    
            Can_set_copy[pos]=L_copy
        else:
            Can_set_copy[pos]=L
    return Can_set_copy
            
            
def store_first_candidate(TS_name,arg1=None):
    
    if arg1==None:
        
        table_dir='Data/'
        can_dir='Candidate/'
        
    elif 'null' in arg1:
        
        r=arg1.split('_')[-1]
        table_dir='处理后/处理后/'+'null_table/Rate_'+r+'/Data/'
        can_dir='处理后/处理后/'+'null_table/Rate_'+r+'/Candidate/'
        with open('Candidate/'+TS_name+'_fcs.json','r',encoding='utf-8') as file:
            OC=json.load(file)
    else:
        
        r=arg1.split('_')[-1]
        table_dir='处理后/处理后/'+'sample_table/Num_'+r+'/Data/'
        can_dir='处理后/处理后/'+'sample_table/Num_'+r+'/Candidate/'
        
    os.makedirs(table_dir,exist_ok=True)
    os.makedirs(can_dir,exist_ok=True)
    with open(table_dir+TS_name+'_table.json','r',encoding='utf-8') as file:
        TableSet=json.load(file)
    with open('Candidate/'+TS_name+'_table_cpl.json','r',encoding='utf-8') as file:
        C=json.load(file)
    #with open('BERT/'+TS_name+'_cbs.json','r',encoding='utf-8') as file:
    #    B=json.load(file)
    
    f=open('../KG/KG_data/type_quantity_classified.data','rb')
    QT=pickle.load(f)
    f.close()
    #can_set_dir='Candidate/'
    #fil_can_dir='处理后/处理后/filter_candidate/'
    model=BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    try:
        with open(can_dir+TS_name+'_fcs.json','r',encoding='utf-8') as file:
            FC=json.load(file)
            print('y')
    except:
        #PC={}
        FC={}
    j=0
    for i in TableSet:
        j+=1
        #if j>3:
        #    break
        if i in FC: 
            if FC[i]!='Fail':
                continue
        print(i,j)
        if 1>0:
            W=TableSet[i]
            if 'null' not in arg1:
                Can_set,td,wd=initial_candidate(W,C[i],model,tokenizer)
                Can_set=preprocess_candidate_type(Can_set,QT)
                #PC[i]=copy.deepcopy(Can_set)
                #write_candidate(Can_set, W, can_set_dir)
                Can_set=candidate_filter_equal_literal(Can_set)
                Can_set=candidate_filter_equal(Can_set)
                #Can_set=traverse_literal_to_entity(Can_set, W, td, wd)
                #Can_set=preprocess_candidate_type(Can_set,QT)
            else:
                Can_set={}
                for pos in OC[i]:
                    
                    row=int(pos.split('_')[0])
                    col=int(pos.split('_')[-1])
                    if W['con'][row][col]=='':
                        Can_set[pos]=[]
                    else:
                        Can_set[pos]=OC[i][pos]
            FC[i]=Can_set
        else:
            #PC[i]='Fail'
            FC[i]='Fail'
    with open(can_dir+TS_name+'_fcs.json','w') as f:
        json.dump(FC,f)
    
    return FC
        
def write_candidate(NCS,WT,fil_can_dir):
    
    f=open(fil_can_dir+WT.name+'_NCS.txt','w',encoding='utf-8')
    for pos in NCS:
        
        f.write('pos\t'+str(pos)+'\n')
        for c in NCS[pos]:
            #print(type(c))
            if type(c)!=str:
                f.write(c.name+'\t'+c['mention']+'\t'+str(c['bs'])+'\t'+str(c['lit'])+'\n')
                f.write('c_type\t')
                for t in c['type']:
                    f.write(t+'\t')
                f.write('\n')
                f.write('c_rin\t')
                for t in c['rin']:
                    f.write(t+'\t')
                f.write('\n')
                f.write('c_rq\t')
                for t in c['rq']:
                    f.write(str(t)+'\t')
                f.write('\n')
            else:
                f.write('qcan\t'+c+'\n')
    f.close()    

def str_to_bool(s):
    return s.lower() == "true"
    
def read_candidate(WT,wt_name,arg1=None):
    

    if arg1==None:
        fil_can_dir='处理后/处理后/filter_candidate/'
    else:
        fil_can_dir=arg1
        os.makedirs(fil_can_dir,exist_ok=True)
    #fil_can_dir='处理后/处理后/filter_candidate/'
    f=open(fil_can_dir+wt_name+'_NCS.txt','r',encoding='utf-8')
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
            #NCS[pos][loc]['rq']=list(filter(lambda x: x != '', NCS[pos][loc]['rq']))
        elif S[0]=='qcan':
            can=Candidate(S[1],WT['con'][pos[0]][pos[1]],[],pos,0,True,[],[])
            loc=len(NCS[pos])
            NCS[pos].append(can)
        else:
            can=Candidate(S[0],S[1],[],pos,float(S[2]),str_to_bool(S[3]),[],[])
            loc=len(NCS[pos])
            NCS[pos].append(can)
    return NCS


if __name__=="__main__":
    #FC=store_first_candidate('T2DC')
    for r in ['0.2','0.3','0.4','0.5']:
        FC=store_first_candidate('T2DC','null_'+r)
    #C=store_candidate_set_literal('T2DC',0.7,False)