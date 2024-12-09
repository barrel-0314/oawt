import json
import csv
import random
from fuzzywuzzy import process
from nltk.corpus import wordnet as wn
import re
from aggregate_query_util import extract_query_keyword
import numpy as np
from scipy.stats import norm
def match_column_refine(match,K):
    
    new_match={}
    col=[]
    for kk in match:
        if isinstance(match[kk],int):
            new_match[kk]=match[kk]
            continue
        if len(match[kk])==1:
            new_match[kk]=match[kk][0]
            col.append(match[kk][0])
    
    for kk in match:
        if kk not in new_match:
            match[kk]=[i for i in match[kk] if i not in col]
            if len(match[kk])==1:
                new_match[kk]=match[kk][0]
                col.append(match[kk][0])
            else:
                min_sum_diff=1000
                for element_x in match[kk]:
                    sum_diff = sum(abs(element_x - new_match[element_y]) for element_y in new_match)  # 计算x中元素与y中所有元素的绝对值差之和
                    if sum_diff < min_sum_diff:
                        min_sum_diff = sum_diff
                        min_element = element_x
                new_match[kk]=min_element
    return new_match

def table_main(Match,K,TS_name,query,lex,rough_flag,sample_tuple,SR_total,table_row):
    #sample_num=200
    non_condition_table_list=[]
    #sample_tuple=[]
    #sample_num_once=100
    sn=0
    #SR_total=[]
    #table_row={}
    #while len(sample_tuple)<sample_num:
    
    for t in Match:
        '''
        if t in non_condition_table_list:
            table_row.pop(t,None)
            continue
        '''
        Match[t]=match_column_refine(Match[t], K)
        #print(Match)
        C=list(csv.reader(open('处理后/处理后/处理后数据集/'+TS_name+'/'+t+'.csv','r',encoding='utf-8')))
        H=C[0]
        content=C[1:]
        if t not in table_row:
            table_row[t]=len(content)-1
        '''
        if len(content)<sample_num_once:
            st=content
        else:
            st=random.sample(content,sample_num_once)
        '''
        st=content
        #print(len(st))
        SR,IN=search_condition(st, K, query, Match[t],lex,rough_flag,H)
        #print(SR,IN,t,len(non_condition_table_list))
        if len(SR)==0:
            non_condition_table_list.append(t)
        
        sample_tuple.extend(st)
        SR_total.extend(SR)
        if len(sample_tuple)!=sn:
            
            sn=len(sample_tuple)
        else:
            break
        if t in non_condition_table_list:
            table_row.pop(t,None)
    sample_num=len(sample_tuple)-IN
    return sample_num,SR_total,table_row,sample_tuple
def sample_table_main(Match,K,TS_name,query,lex,rough_flag,sample_tuple,SR_total,table_row):
    
    sample_num=200
    non_condition_table_list=[]
    #sample_tuple=[]
    sample_ite={}
    sample_num_once=100
    sn=0
    #SR_total=[]
    #table_row={}
    IN=0
    ite=0
    while len(sample_tuple)<sample_num:
        if ite>100:
            break
        #print(len(sample_tuple))
        if len(non_condition_table_list)==len(Match):
            break
        for t in Match:
            if t in non_condition_table_list:
                table_row.pop(t,None)
                continue
            Match[t]=match_column_refine(Match[t], K)
            #print(Match)
            C=list(csv.reader(open('处理后/处理后/处理后数据集/'+TS_name+'/'+t+'.csv','r',encoding='utf-8')))
            H=C[0]
            content=C[1:]
            con_sam=[row for row in content if row not in sample_tuple]
            if t not in table_row:
                table_row[t]=len(content)-1
            if len(con_sam)<sample_num_once:
                st=con_sam
            else:
                st=random.sample(con_sam,sample_num_once)
            #print(len(st[0]),'st')
            if len(st)==0:
                #non_condition_table_list.append(t)
                break
            SR,IN=search_condition(st, K, query, Match[t],lex,rough_flag,H)
            #print(SR,IN,t,len(non_condition_table_list))
            if len(SR)==0:
                #print('b')
                break
            sample_tuple.extend(st)
            SR_total.extend(SR)
            
            if len(sample_tuple)!=sn:
                
                sn=len(sample_tuple)
            else:
                break
        if len(st)==0:
            break
        ite+=1
    sample_num=len(sample_tuple)-IN
    return sample_num,SR_total,table_row,sample_tuple
            
def extract_first_number(s):
    match = re.search(r'\b(?:\d+\.\d*|\.\d+|\d+)\b', s)
    #print(s,match)
    if match:
        return match.group()
    else:
        return None  
        
def search_condition(sample_tuple,K,query,match,lex,rough_flag,H):
    
    invalid_num=0
    if '<' in query['WHERE']:
        k=query['WHERE'].split('<')
        #print(len(sample_tuple[0]),'l')
        #print(match,match[k[0]])
        N=[]
        L=[]
        if 'million' in H[match[k[0]]].lower():
            ti=10**6
        elif 'billion' in H[match[k[0]]].lower():
            ti=10**9
        else:
            ti=1
        for s in sample_tuple:
            try:
                N.append(float(extract_first_number(s[match[k[0]]].replace(',','')))*ti)
                L.append(sample_tuple.index(s))
            except:
                invalid_num+=1
        #N=[extract_first_number(s[match[k[0]]].replace(',','')) for s in sample_tuple if len(s)>match[k[0]]]
        satisfied_row=[]
        #invalid_num=0
        
        for i in range(len(L)):
            
            if float(N[i])<float(k[1]):
                #print(N[i],sample_tuple[i])
                t=[sample_tuple[L[i]][match[kk]] for kk in K if kk!='']
                #loc=K.index(k[0])
                #t[loc]=N[i]
                t.append(ti)
                satisfied_row.append(t)
            
        
    elif '>' in query['WHERE']:
        
        k=query['WHERE'].split('>')
        N=[]
        L=[]
        if 'million' in H[match[k[0]]].lower():
            ti=10**6
        elif 'billion' in H[match[k[0]]].lower():
            ti=10**9
        else:
            ti=1
        for s in sample_tuple:
            try:
                N.append(float(extract_first_number(s[match[k[0]]].replace(',','')))*ti)
                L.append(sample_tuple.index(s))
            except:
                invalid_num+=1
        satisfied_row=[]
        #invalid_num=0
        
        for i in range(len(L)):
            
            if float(N[i])>float(k[1]):
                t=[sample_tuple[L[i]][match[kk]] for kk in K if kk!='']
                t.append(ti)
                satisfied_row.append(t)
    elif '=' in query['WHERE']:
         
         k=query['WHERE'].split('=')
         entity=k[-1]
         #print(k[-1])
         satisfied_row=[]
         if K[1]!='':
             if 'million' in H[match[K[1]]].lower():
                 ti=10**6
             elif 'billion' in H[match[K[1]]].lower():
                 ti=10**9
             else:
                 ti=1
         else:
             if 'million' in H[match[K[0]]].lower():
                 ti=10**6
             elif 'billion' in H[match[K[0]]].lower():
                 ti=10**9
             else:
                 ti=1
         if K[-1]!='':
             #print(K[-1])
             tcol=[i[match[K[-1]]] for i in sample_tuple]
             #print(tcol,entity)
             res=rough_search(tcol,entity,lex,rough_flag)
             #print(res)
             if len(res)!=0:
                 
                 for i in range(len(res)):
                     for j in range(len(sample_tuple)):
                         
                         if res[i] ==sample_tuple[j][match[K[-1]]]:
                             t=[sample_tuple[j][match[kk]] for kk in K if kk!='']
                             t.append(ti)
                             satisfied_row.append(t)
                         
         else:
            
            tcol=[i[match[K[0]]] for i in sample_tuple]
            res=rough_search(tcol,entity,lex,True)
            if len(res)!=0:
                
                for i in range(len(res)):
                    for j in range(len(sample_tuple)):
                        
                        if res[i] ==sample_tuple[j][match[K[0]]]:
                            t=[sample_tuple[j][match[kk]] for kk in K if kk!='']
                            t.append(ti)
                            satisfied_row.append(t)
    else:
        satisfied_row=[]
        if K[1]!='':
            if 'million' in H[match[K[1]]].lower():
                ti=10**6
            elif 'billion' in H[match[K[1]]].lower():
                ti=10**9
            else:
                ti=1
        else:
            if 'million' in H[match[K[0]]].lower():
                ti=10**6
            elif 'billion' in H[match[K[0]]].lower():
                ti=10**9
            else:
                ti=1
        '''
        if query['WHERE'].split('.')[-1]!='':
            loc=K[1]
        else:
            loc=K[0]
        '''
        for j in range(len(sample_tuple)):
            t=[sample_tuple[j][match[kk]] for kk in K if kk!='']
            t.append(ti)
            satisfied_row.append(t)
            #satisfied_row=[extract_first_number(i[match[loc]]) for i in sample_tuple if i[match[loc]] is not None]
        #print(len(satisfied_row),'len')
    return satisfied_row,invalid_num
    
    
def rough_search(tcol,entity,lex,rough_flag):

    S=[]    
    #s=[j[0] for j in process.extract(entity,tcol) if j[1]>90]
    Syn=[]
    for synset in wn.synsets(entity):
        for lemma in synset.lemmas():
            Syn.append(lemma.name())
    if entity not in Syn:
        Syn.append(entity)
    #Syn=set([str(w).split('.')[0].split('(')[-1] for w in wn.synsets(entity)]+[entity])
    #print(Syn)
    if rough_flag==False:
        Syn=[entity]
    for sy in Syn:
        
        s=[j[0] for j in process.extract(sy,tcol) if j[1]>lex]
        S=S+s
        
    S=list(set(s))
    return S

def extract_num_mention(satisfied_row,loc):
    
    num_array=[]
    for s in satisfied_row:
        try:
            #print(s)
            #print(extract_first_number(s[loc].replace(',','')))
            num_array.append(s[-1]*float(extract_first_number(s[loc].replace(',',''))))
        except:
            continue
    return num_array

def calculate_op_result(query,satisfied_row,table_row,sample_num,K):
    
    op=query['SELECT'].split('(')[0]
    #print(len(satisfied_row),'st')
    if len(satisfied_row)==0:
        estimate_result=0
        num_array=np.zeros(len(satisfied_row))
        estimate_result=0
    elif op=='COUNT':
        total_num=np.sum(np.array([table_row[t] for t in table_row]))
        #print([table_row[t] for t in table_row])
        #sample_num=len(sample_tuple)-invalid_num
        num_array=np.zeros(len(satisfied_row))
        estimate_result=(total_num/sample_num)*len(satisfied_row)
        #print(estimate_result,total_num,sample_num,'er')
    elif op=='SUM':
        if '>' in query['WHERE']:
            k=query['WHERE'].split('>')[0]
        elif '<' in query['WHERE']:
            k=query['WHERE'].split('<')[0]
        elif '=' in query['WHERE']:
            k=K[1]
        else:
            if query['WHERE'].split('.')[-1]!='':
                k=K[1]
            else:
                k=K[0]
            #k=K[0]
        total_num=np.sum(np.array([table_row[t] for t in table_row]))
        #sample_num=len(satisfied_row)
        
        num_array=np.array(extract_num_mention(satisfied_row, K.index(k)))
        sum_value=np.sum(np.array(num_array))
        estimate_result=(total_num/sample_num)*sum_value
    elif op=='AVG':
        if '>' in query['WHERE']:
            k=query['WHERE'].split('>')[0]
        elif '<' in query['WHERE']:
            k=query['WHERE'].split('<')[0]
        elif '=' in query['WHERE']:
            k=K[1]
        else:
            if query['WHERE'].split('.')[-1]!='':
                k=K[1]
            else:
                k=K[0]
        total_num=np.sum(np.array([table_row[t] for t in table_row]))
        #sample_num=len(sample_tuple)-invalid_num
        count_value=(total_num/sample_num)*len(satisfied_row)
        num_array=np.array(extract_num_mention(satisfied_row, K.index(k)))
        sum_value=(total_num/sample_num)*np.sum(np.array(num_array))
        #print(len(num_array),len(satisfied_row),k,'na')
        estimate_result=sum_value/count_value
    '''
    elif op=='MAX':
        if '>' in query['WHERE']:
            k=query['WHERE'].split('>')[0]
        elif '<' in query['WHERE']:
            k=query['WHERE'].split('<')[0]
        elif '=' in query['WHERE']:
            k=K[1]
        else:
            k=K[0]
        num_array=extract_num_mention(satisfied_row, K.index(k))
        if len(num_array)==0:
            estimate_result=0
        else:
            estimate_result=max(num_array)
    elif op=='MIN':
        if '>' in query['WHERE']:
            k=query['WHERE'].split('>')[0]
        elif '<' in query['WHERE']:
            k=query['WHERE'].split('<')[0]
        elif '=' in query['WHERE']:
            k=K[1]
        else:
            k=K[0]
        num_array=extract_num_mention(satisfied_row, K.index(k))
        if len(num_array)==0:
            estimate_result=0
        else:
            estimate_result=min(num_array)
    '''
    return estimate_result,num_array,op

def confidence_interval_sample(estimate_result,table_row,sample_num,num_array,op,satisfied_row,epsilon):
    
    #epsilon=estimate_result*error/(1+error)
    #print(len(num_array))
    total_num=np.sum(np.array([table_row[t] for t in table_row]))
    if sample_num==0:
        sample_num=1
        
    if op=='COUNT':
        #print(estimate_result,total_num)
        sigma=(estimate_result/sample_num*(total_num**2)-estimate_result**2)**(1/2)
        #print()
    elif op=='SUM':
        
        sigma=(np.sum(num_array**2)/sample_num*(total_num**2)-estimate_result**2)**(1/2)
    
    elif op=='AVG':
        
        count_sample=(total_num/sample_num)*len(satisfied_row)
        #print(total_num,sample_num,len(num_array))
        sigma=(np.sum(num_array**2)/count_sample*(total_num**2)-estimate_result**2)**(1/2)
    
    elif op=='MAX':
        
        sigma=(len(num_array)*(estimate_result**2)/count_sample*(total_num**2)-estimate_result**2)**(1/2)
    
    elif op=='MIN':
        
        sigma=(len(num_array)*(estimate_result**2)/count_sample*(total_num**2)-estimate_result**2)**(1/2)
    mu=0
    sigma=sigma/(sample_num**(1/2))
    '''
    sig=sigma/(sample_num**(1/2))
    mu=0
    cdfp=norm.cdf(epsilon,loc=mu,scale=sig)
    P=2*cdfp-1
    '''
    return mu,sigma
        
def confidence_interval_noise(estimate_result,table_row,sample_num,num_array,op,satisfied_row,epsilon,pro):
    
    #epsilon=estimate_result*error/(1+error)
    total_num=np.sum(np.array([table_row[t] for t in table_row]))
    if op=='COUNT':
        
        mu=pro*len(num_array)*total_num/sample_num
        sigma=(pro*(1-pro)*len(num_array)*(total_num/sample_num)**2)**(1/2)
        
    elif op=='SUM':
        
        mu=pro*(total_num/sample_num)*np.sum(num_array)
        sigma=pro*(1-pro)*(np.sum(num_array**2)*(total_num/sample_num)**2)**(1/2)
    
    elif op=='AVG':
        
        count_sample=(total_num/sample_num)*len(satisfied_row)
        mu=pro*(total_num/(sample_num*len(num_array)))*np.sum(num_array)
        sigma=(pro*(1-pro)*np.sum(num_array**2)*(total_num/count_sample)**2)**(1/2)
    
    return mu,sigma

def pro_cdf(mu,sigma,error,estimate_result,sample_num):

    epsilon=estimate_result*error/(1+error)
    sig=sigma/(sample_num**(1/2))
    mu=0
    cdfp=norm.cdf(epsilon,loc=mu,scale=sig)
    P=2*cdfp-1
    return P
    
if __name__=="__main__":
    
    TS_name='SemTab'
    with open('Query/'+TS_name+'_aq.json','r',encoding='utf-8') as f:
        Q=json.load(f)
    with open('Query/'+TS_name+'_match.json','r',encoding='utf-8') as f:
        M=json.load(f)
    i=92
    q='Query '+str(i)
    query=Q[q]
    print(Q[q])
    k1,k2,k3=extract_query_keyword(query)
    if k3=='countries':
        k3='country'
    K=[k1,k2,k3]
    SN,SR,TR=table_main(M[q],K,TS_name,query,70,True)
    ER=calculate_op_result(query,SR,TR,SN,K)