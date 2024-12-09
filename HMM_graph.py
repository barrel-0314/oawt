import networkx as nx
import numpy as np
from quantity_candidate_generate import difference_quantity_candidate

def construct_graph(NCS,WT,ctf,ctfq,ce,cq):
    
    GL={}
    for c in WT['qcol']:
        
        if int(c) in cq:
            
            qcol=ce[cq.index(int(c))]
            
        else:
            
            qcol=int(WT['core'])
        G=nx.DiGraph()
        G=add_mention(WT,G,qcol,int(c))
        G=add_candidate(WT,NCS,G,qcol,int(c))
        G=add_column_type(ctf,WT,G,qcol,NCS)
        G=add_quantity_type(ctfq,WT,G,int(c),NCS)
        #print(list(G.nodes()))
        G=add_other_edges(G,WT,qcol,int(c),ctfq,ctf)
        G=add_weighted_edge(G,qcol,int(c),WT,NCS,ctf,ctfq)
        G=dict_weight_graph(G)
        #nx.draw(G,with_labels=True)
        #plt.show()
        #print(len(G.nodes()))
        GL[int(c)]=G
        
    return GL
    
def add_mention(WT,G,qcol,c):
    
    for i in range(len(WT['con'])):
        
        for j in range(WT['col_num']):
            
            if j==qcol or j==c:
                if j>=len(WT['con'][i]):
                    m=WT['con'][i][-1]
                else:
                    m=WT['con'][i][j]
                G.add_node((i,j,m,'men'))
                G.nodes[(i,j,m,'men')]['nt']='mention'
        if qcol>=len(WT['con'][i]):
            em=WT['con'][i][-1]
        else:
            em=WT['con'][i][qcol]
        if c>=len(WT['con'][i]):
            qm=WT['con'][i][-1]
        else:
            qm=WT['con'][i][c]
        G.add_edge((i,qcol,em,'men'), (i,c,qm,'men'))
        G.edges[(i,qcol,em,'men'),(i,c,qm,'men')]['w']=1 #the same row mention edge
        G.add_edge((i,c,qm,'men'),(i,qcol,em,'men'))
        G.edges[(i,c,qm,'men'),(i,qcol,em,'men')]['w']=1 #the same row mention edge

    return G
    
def add_candidate(WT,NCS,G,qcol,c):
    
    
    for pos in NCS:
        #bert_s_mx=max(can['bs'] for can in NCS[pos])
        p0=int(pos.split('_')[0])
        p1=int(pos.split('_')[1])
        if p1==qcol:
        
            for k in range(len(NCS[pos])):
                
                can=NCS[pos][k]
                G.add_node((p0,p1,k,'can'))
                G.nodes[(p0,p1,k,'can')]['nt']='candidate'
                m=WT['con'][p0][p1]
                G.add_edge((p0,p1,m,'men'),(p0,p1,k,'can'))
                G.add_edge((p0,p1,k,'can'),(p0,p1,m,'men'))
                '''
                if p1==c:
                    G.edges[(p0,p1,m),(p0,p1,k,can)]['w']=(1/len(NCS[pos])) #mention-quantity candidate edge
                    G.edges[(p0,p1,k,can),(p0,p1,m)]['w']=1 #quantity candidate-mention edge
                '''
                #else:
                G.edges[(p0,p1,m,'men'),(p0,p1,k,'can')]['w']=can['bs'] #mention-entity candidate edge
                G.edges[(p0,p1,k,'can'),(p0,p1,m,'men')]['w']=1 #entity candidate-mention edge
            
    return G
    
def add_column_type(ctf,WT,G,qcol,NCS):
    
    for k in range(len(ctf[str(qcol)])):
        
        t=ctf[str(qcol)][k]
        G.add_node((qcol,k,t['name'],'et'))
        #print(t['t_name'])
        G.nodes[(qcol,k,t['name'],'et')]['nt']='entityType'
        for pos in NCS:
            p0=int(pos.split('_')[0])
            p1=int(pos.split('_')[1])
            if p1==qcol:
                
                CL=NCS[pos]
                #print(CL,'cl')
                for can in CL:
                    #print(can['type'],can.rin)
                    if t['name'] in can['type'] or t['name'] in can['rin']:
                        #print(t)
                        G.add_edge((qcol,k,t['name'],'et'),(p0,p1,CL.index(can),'can'))
                        G.add_edge((p0,p1,CL.index(can),'can'),(qcol,k,t['name'],'et'))
                        G.edges[(p0,p1,CL.index(can),'can'),(qcol,k,t['name'],'et')]['w']=1 #entity candidate-column type edge
                        G.edges[(qcol,k,t['name'],'et'),(p0,p1,CL.index(can),'can')]['w']=-1 #column type-entity candidate edge
    return G                        

        
    
def add_quantity_type(ctfq,WT,G,c,NCS):

    for k in range(len(ctfq[c])):
        
        t=ctfq[c][k]
        
        if t['ic']!=-1:
            #s=t['ts']
            G.add_node((c,k,t['name'],'qt'))
            G.nodes[(c,k,t['name'],'qt')]['nt']='quantityType'
            #rint(t,t['t_name'],'qt')
        else:
            G.add_node((c,k,t['name'],'qh'))
            G.nodes[(c,k,t['name'],'qh')]['nt']='quantityHeader'
            #print(t,t['t_name'],'qh')
        '''
        for pos in NCS:
            
            if p1==c:
                
                CL=NCS[pos]
                for can in CL:
                    
                    #if t['t_name'] in can['type'] or t['t_name'] in can.rin:
                    #print('t')
                    G.add_edge((c,k,t),(p0,p1,CL.index(can),can)) #column type-quantity mention edge
                    G.edges[(c,k,t),(p0,p1,CL.index(can),can)]['w']=-1
                    #print(t['t_name'],can.name)
                    G.add_edge((p0,p1,CL.index(can),can),(c,k,t)) #quantity mention-column type edge
                    G.edges[(p0,p1,CL.index(can),can),(c,k,t)]['w']=1
                    #print(t['t_name'],can.name)
        '''
    return G                    

def add_other_edges(G,WT,qcol,c,ctfq,ctf):
    
    
    nodes_mention = [node for node, attrs in G.nodes(data=True) if attrs['nt']=='mention']
    
    for node in nodes_mention:
        
        nei=list(G.neighbors(node))
        flag=0
        for n in nei:
            
            if G.nodes[n]['nt']=='candidate':
                
                flag=1
                break
        if flag==0:
            #print(node)
            if node[1]==qcol:
                
                for i in range(len(ctf[str(qcol)])):
                
                    G.add_edge(node,(qcol,i,ctf[str(qcol)][i]['name'],'et')) #entity mention-column type edge
                    G.edges[node,(qcol,i,ctf[str(qcol)][i]['name'],'et')]['w']=ctf[str(qcol)][i]['bs']
                    G.add_edge((qcol,i,ctf[str(qcol)][i]['name'],'et'),node) #column type-entity mention edge
                    G.edges[(qcol,i,ctf[str(qcol)][i]['name'],'et'),node]['w']=-1

            else:
                for i in range(len(ctfq[c])):
                
                    
                    if ctfq[c][i]['ic']!=-1:
                        G.add_edge(node,(c,i,ctfq[c][i]['name'],'qt')) #quantity mention-column type edge
                        G.add_edge((c,i,ctfq[c][i]['name'],'qt'),node) #column type-quantity mention edge
                        #print(ctfq[c][i]['ts'],node)
                        G.edges[node,(c,i,ctfq[c][i]['name'],'qt')]['w']=ctfq[c][i]['ts'][node[0]]
                        G.edges[(c,i,ctfq[c][i]['name'],'qt'),node]['w']=-1
                        #print(ctfq[c][i]['ts'][node[0]],'w')
                    else:
                        G.add_edge(node,(c,i,ctfq[c][i]['name'],'qh')) #quantity mention-column type edge
                        G.add_edge((c,i,ctfq[c][i]['name'],'qh'),node) #column type-quantity mention edge
                        G.edges[node,(c,i,ctfq[c][i]['name'],'qh')]['w']=ctfq[c][i]['ts']
                        #print(ctfq[c][i]['bs'],'w')
                        G.edges[(c,i,ctfq[c][i]['name'],'qh'),node]['w']=-1
    '''
    for r in range(len(WT['con'])):
        
        me=WT['con'][r][qcol]
        mq=WT['con'][r][c]
        
        for rho in range(r+1,len(WT['con'])):
            
            me_rho=WT['con'][rho][qcol]
            mq_rho=WT['con'][rho][c]
            G.add_edge((r,qcol,me),(rho,qcol,me_rho))
            G.edges[(r,qcol,me),(rho,qcol,me_rho)]['w']=(1/len(WT['con'])) #the same column entity mention
            G.add_edge((r,qcol,me),(rho,qcol,me_rho))
            G.edges[(r,qcol,me),(rho,qcol,me_rho)]['w']=(1/len(WT['con'])) #the same column entity mention
            
            G.add_edge((rho,c,mq_rho),(r,c,mq))
            G.edges[(rho,c,mq_rho),(r,c,mq)]['w']=(1/len(WT['con'])) #the same column quantity mention
            G.add_edge((rho,c,mq_rho),(r,c,mq))
            G.edges[(rho,c,mq_rho),(r,c,mq)]['w']=(1/len(WT['con'])) #the same column quantity mention
    '''        
    return G    
    
def add_weighted_edge(G,qcol,c,WT,NCS,ctf,ctfq): 
    
    nodes_type = [node for node, attrs in G.nodes(data=True) if attrs['nt']=='entityType' or 'quantityType']
    #print(len(G.nodes()))
    for node in nodes_type:
        
        nei=list(G.neighbors(node))
        '''
        if True in [G.nodes[n]['nt']=='candidate' for n in nei]:
            
            flag=1
            
        else:
            
            flag=0
        '''
        for n in nei:
            
            #print(G.edges[node,n])
            if G.edges[node,n]['w']==-1:
                
                G.edges[node,n]['w']=1/len(nei)
                '''
                if G.nodes[n]['nt']=='candidate':
                    
                    G.edges[node,n]['w']=1/(len(NCS[n[0],n[1]])*len(WT['con']))
                    
                elif G.nodes[n]['nt']=='mention' and flag==1:
                    
                    G.edges[node,n]['w']=1/len(WT['con'])
                    
                elif G.nodes[n]['nt']=='mention' and flag==0:
                    
                    G.edges[node,n]['w']=1/len(nei)
                '''
                
    nodes_mention = [node for node, attrs in G.nodes(data=True) if attrs['nt']=='mention']
    
    for node in nodes_mention:
        #print(node)
        nei=[nd for nd in list(G.neighbors(node)) if G.nodes[nd]['nt']=='candidate' and nd[1]==qcol]
        b_nei=np.array([-NCS[str(can[0])+'_'+str(can[1])][can[2]]['bs'] for can in nei])
        softmax_b_nei=function_for_normalize(b_nei)
        #print(type(softmax_b_nei))
        #softmax_b_nei = np.exp(b_nei) / np.sum(np.exp(b_nei))
        for n in nei:
            
            G.edges[node,n]['w']=softmax_b_nei[nei.index(n)]
        
        nei=[nd for nd in list(G.neighbors(node)) if G.nodes[nd]['nt']=='candidate' and nd[1]==c]
        b_nei=np.array([-NCS[str(can[0])+'_'+str(can[1])][can[2]]['bs'] for can in nei])
        softmax_b_nei=function_for_normalize(b_nei)
        #softmax_b_nei = np.exp(b_nei) / np.sum(np.exp(b_nei))
        for n in nei:
            
            G.edges[node,n]['w']=softmax_b_nei[nei.index(n)]
            
        nei=[nd for nd in list(G.neighbors(node)) if G.nodes[nd]['nt']=='entityType' and nd[0]==qcol]
        #print(nei)
        if len(nei)!=0:
            #print(type(nei[0][0]))
            b_nei=np.array([-ctf[str(can[0])][can[1]]['bs'] for can in nei])
            softmax_b_nei=function_for_normalize(b_nei)
            #softmax_b_nei = np.exp(b_nei) / np.sum(np.exp(b_nei))
            for n in nei:
                
                G.edges[node,n]['w']=softmax_b_nei[nei.index(n)]
            #print(b_nei,softmax_b_nei)
        nei=[nd for nd in list(G.neighbors(node)) if G.nodes[nd]['nt']=='quantityType' and nd[0]==c]
        #print(nei)
        #print(len(nei),'qt')
        if len(nei)!=0:
            #print([-ctfq[can[0]][can[1]]['ts'][node[0]] for can in nei])
            b_nei=np.array([-ctfq[can[0]][can[1]]['ts'][node[0]] for can in nei])
            
            #print(type(b_nei))
            #print(b_nei.shape)
            softmax_b_nei1=function_for_normalize(b_nei)
            #softmax_b_nei1 = np.exp(b_nei) / np.sum(np.exp(b_nei))
            h_nei=np.array([(1-ctfq[can[0]][can[1]]['hs']) for can in nei])
            #print(h_nei.shape)
            softmax_b_nei_h=np.exp(h_nei) / np.sum(np.exp(h_nei))
            #print(softmax_b_nei_h.shape)
            #print(G.nodes[node]['nt'],[nn[-1]['t_name'] for nn in nei],'nn')
            if str(node[0])+'_'+str(qcol) in NCS and NCS[str(node[0])+'_'+str(qcol)]!=[]:
                if node[1]>=len(WT['con'][node[0]]):
                    softmax_b_nei2=add_weight_on_candidate_to_type(NCS[str(node[0])+'_'+str(qcol)][0], WT['con'][node[0]][-1], nei)            
                else:
                    softmax_b_nei2=add_weight_on_candidate_to_type(NCS[str(node[0])+'_'+str(qcol)][0], WT['con'][node[0]][node[1]], nei)            
                softmax_b_nei=(softmax_b_nei1+softmax_b_nei_h)*softmax_b_nei2
                softmax_b_nei=function_for_normalize(softmax_b_nei)
                #softmax_b_nei=np.exp(softmax_b_nei) / np.sum(np.exp(softmax_b_nei))
                #print(softmax_b_nei1,'b1',softmax_b_nei2,softmax_b_nei_h)
            else:#去掉与quantityType的连接，放大与mention的权重
                #softmax_b_nei=softmax_b_nei1+softmax_b_nei_h
                #softmax_b_nei=function_for_normalize(softmax_b_nei)
                #softmax_b_nei=np.exp(softmax_b_nei) / np.sum(np.exp(softmax_b_nei))
                #print(softmax_b_nei_h,softmax_b_nei1)
                #print(softmax_b_nei1,softmax_b_nei_h)
                softmax_b_nei=np.zeros(len(nei))
                #print(len(list(G.neighbors(node))))
                #for n in list(G.neighbors(node)):
                #    G.edges[node,n]['w']*=2
                #    print('y')
                    
            for n in nei:
                G.edges[node,n]['w']=softmax_b_nei[nei.index(n)]
                    
            #print(np.sum(softmax_b_nei))
        nei=[nd for nd in list(G.neighbors(node)) if G.nodes[nd]['nt']=='quantityHeader' and nd[0]==c]
        #print(nei)
        #print(len(nei),'qh')
        if len(nei)!=0:
            #print([can[-1]['bs'] for can in nei])
            b_nei=np.array([-ctfq[can[0]][can[1]]['ts'][node[0]]/100 for can in nei])
    
            softmax_b_nei = np.exp(b_nei) / np.sum(np.exp(b_nei))
            for n in nei:
                
                G.edges[node,n]['w']=softmax_b_nei[nei.index(n)]
            #print(b_nei,softmax_b_nei)
    
    for node in nodes_mention:
        
        if node[1]==qcol:
            
            if True not in [G.nodes[n]['nt']=='entityType' for n in list(G.neighbors(node))] and True not in [G.nodes[n]['nt']=='candidate' for n in list(G.neighbors(node))]:
                
                nei=list(G.neighbors(node))
                #print(len(nei),'fin')
                for n in nei:
                    
                    G.edges[node,n]['w']*=2
    
    nodes_can = [node for node, attrs in G.nodes(data=True) if attrs['nt']=='candidate']
    
    for node in nodes_can:
        
        if node[1]==qcol:
            nei=[nd for nd in list(G.neighbors(node)) if G.nodes[nd]['nt']=='entityType' and nd[0]==qcol]
            #print(nei)
            for n in nei:
                
                G.edges[node,n]['w']=1/len(nei)
        '''
        else:
            nei=[nd for nd in list(G.neighbors(node)) if G.nodes[nd]['nt']=='quantityType' and nd[0]==c]
            #print(nei,node)
            R,softmax_b_nei=add_weight_on_candidate_to_type(node[-1], WT['con'][node[0]][node[1]], nei)
            print(R)
            if R==[]:
                for n in nei:
                    #print(n[-1]['t_name'])
                    G.edges[node,n]['w']=1/len(nei)
            else:
                for n in nei:
                    #print(n[-1]['t_name'])
                    if n[-1]['t_name'] in R:
                        G.edges[node,n]['w']=softmax_b_nei[R.index(n[-1]['t_name'])]
                    else:
                        G.remove_edge(node,n)
        '''
    '''
    for u,v in G.edges():
        
        print(u,v,G.edges[u,v])
    '''
    return G
    
def function_for_normalize(N):
    
    if np.sum(N)!=0:
        N_normal=N/np.sum(N)
    else:
        N_normal=np.exp(N) / np.sum(np.exp(N))
    return N_normal

def add_weight_on_candidate_to_type(can,mention,nei):
    
    R=[]
    Q=[]
    T=[n[2] for n in nei]
    #print(can.rq,'rq')
    for tur in can['rq']:
        
        rel=tur[0]#[2:-2]
        q=tur[1]#.replace(' ','')
        #q=q[2:-2]
        #print(q,'q')
        #print(rel,q,mention)
        if rel in T:
            try:
                s=difference_quantity_candidate(q, mention)
            except:
                s=float('inf')
            R.append(rel)
            Q.append(-s)
    #print(R,Q,mention)   
    #print(Q,'q')     
    if len(Q)==0:
        b_nei=np.array(Q)
        #print(b_nei,'b')
    elif min(Q)<-100 or max(Q)>100:
        
        if len(set(Q))==1 and abs(min(Q))==float('inf'):
            b_nei=np.zeros(len(Q))
        else:
            b_nei=np.array(Q)/abs(min(Q))
            for n in range(len(Q)):
                
                if np.isnan(b_nei[n]):
                    
                    b_nei[n]=1
            #print(b_nei,'b')
    else:
        b_nei=np.array(Q)

    softmax_b_nei = np.exp(b_nei) / np.sum(np.exp(b_nei))   
    #print(b_nei,softmax_b_nei)
    QN=np.zeros(len(nei))
    #QN[0:len(R)]=softmax_b_nei
    i=0
    for n in nei:
        
        if n[2] in R:
            #print(R[R.index(n[-1]['t_name'])])
            QN[i]=softmax_b_nei[R.index(n[2])]
        i+=1    
    return QN


def dict_weight_graph(G):
    
    P={}
    P['mention-mention']=0.5
    P['mention-candidate']=0.5
    P['mention-type']=0.5
    P['candidate-mention']=0.5
    P['candidate-type']=0.5
    P['type-mention']=1
    P['type-candidate']=1
    
    for u,v in G.edges():
        
        if G.nodes[u]['nt']=='mention' and G.nodes[v]['nt']=='mention':
                        
            G.edges[u,v]['w']=P['mention-mention']*G.edges[u,v]['w']
            
        elif G.nodes[u]['nt']=='mention' and G.nodes[v]['nt']=='candidate':
            
            G.edges[u,v]['w']=P['mention-candidate']*G.edges[u,v]['w']
        
        elif G.nodes[u]['nt']=='mention' and G.nodes[v]['nt']=='quantityType' or G.nodes[v]['nt']=='quantityHeader':
            
            G.edges[u,v]['w']=P['mention-type']*G.edges[u,v]['w']
            
        elif G.nodes[u]['nt']=='mention' and G.nodes[v]['nt']=='entityType':
            
            G.edges[u,v]['w']=P['mention-type']*G.edges[u,v]['w']

        elif G.nodes[u]['nt']=='candidate' and G.nodes[v]['nt']=='entityType':
            
            G.edges[u,v]['w']=P['candidate-type']*G.edges[u,v]['w']    
            
        elif G.nodes[u]['nt']=='candidate' and G.nodes[v]['nt']=='quantityType':
            
            G.edges[u,v]['w']=P['candidate-type']*G.edges[u,v]['w']    
            
        elif G.nodes[u]['nt']=='candidate' and G.nodes[v]['nt']=='mention':
            
            G.edges[u,v]['w']=P['candidate-mention']*G.edges[u,v]['w']
            
        elif G.nodes[u]['nt']=='quantityType' and G.nodes[v]['nt']=='mention':
            
            G.edges[u,v]['w']=P['type-mention']*G.edges[u,v]['w']
            
        elif G.nodes[u]['nt']=='entityType' and G.nodes[v]['nt']=='mention':
            
            G.edges[u,v]['w']=P['type-mention']*G.edges[u,v]['w']

        elif G.nodes[u]['nt']=='quantityType' and G.nodes[v]['nt']=='candidate':
            
            G.edges[u,v]['w']=P['type-candidate']*G.edges[u,v]['w']
            
        elif G.nodes[u]['nt']=='entityType' and G.nodes[v]['nt']=='candidate':
            
            G.edges[u,v]['w']=P['type-candidate']*G.edges[u,v]['w']
    
    
    for node in G.nodes():
        
        nei=list(G.neighbors(node))
        s=0
        for n in nei:
            #print(G.nodes[node]['nt'],G.nodes[n]['nt'],G.edges[node,n]['w'])
            #if G.edges[node,n]['w']==np.nan:
                #print(G.nodes[node]['nt'],G.nodes[n]['nt'],G.edges[node,n]['w'])

            s+=G.edges[node,n]['w']
        #print(s)
        '''
        if s<0.99 or s>1.01:
            print(node,len(nei))
            for n in nei:
                G.edges[node,n]['w']=G.edges[node,n]['w']/s
                print(node,n)
                print(G.nodes[node]['nt'],G.nodes[n]['nt'],G.edges[node,n]['w'])
            print('t',s)        
        '''
    return G
    