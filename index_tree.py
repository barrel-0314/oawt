import networkx as nx
import sys
import math
import pickle
import time
from rdflib import Literal,Namespace

class Node:
    def __init__(self, scala = 3):
        self.values = []
        self.parent = None
        self.children = []
 
    def insert(self, value):
        if value in self.values:
            pass
        else:
            self.values.append(value)
            self.values.sort()
        return len(self.values)
    
    def compare(self, value):
        length = len(self.values)
        if self.children == [] or value in self.values:
            return None
        
        for i in range(length):
            if value < self.values[i]:
                return i
        return i + 1    
 
    def getPos(self):
        return self.parent.children.index(self)
    
    def getValLen(self):
        return len(self.values)   
 
class BTree:
    def __init__(self, node:Node = None, scala = 3):
        self.root = Node(scala = scala)
        self.scala = scala
        self.mid_index = int((self.scala-1)/2)
 
    def _find(self, value, node:Node = None):
        if not node:
            return BTree.compare(value, self.root)
        else:
            return BTree.compare(value, node)
    
    def find(self, value, node:Node = None):
        if not node:
            _node = self.root
        else:
            _node = node
        
        result = _node.compare(value)
        if result == None:
            return _node
        else:
            return self.find(value, node = _node.children[result])
            
    def _split(self, node):
        if len(node.values) <= self.scala - 1:
            return 0
 
        parent = node.parent
        new_node, l_node, r_node = Node(),Node(), Node()
        
        mid_index = self.mid_index
        l_node.values = node.values[0:mid_index]
        center = node.values[mid_index]
        r_node.values = node.values[mid_index+1:]
 
        if node.children != []:
            l_node.children = node.children[0:mid_index+1]
            r_node.children = node.children[mid_index+1:]
            for i in range(mid_index+1):
                node.children[i].parent = l_node
            for i in range(mid_index+1, self.scala + 1):
                node.children[i].parent = r_node
 
        if not parent:
            parent = new_node
            parent.values.append(center)
            parent.children.insert(0, l_node)
            parent.children.insert(1, r_node)
            l_node.parent = parent
            r_node.parent = parent
            self.root = parent
            return 0
        
        l_node.parent = parent
        r_node.parent = parent
        parent.insert(center)
        index = parent.children.index(node)
        parent.children.pop(index)
        parent.children.insert(index, l_node)
        parent.children.insert(index + 1, r_node)
        return self._split(parent)
 
    def stepCover(self, node:Node, value_pos):     # value_pos表示删除的value所在的位置
        if node.children == []:
            return self.merge(node, node.getPos())
 
        after = node.children[value_pos + 1]
        node.insert(after.values.pop(0))
        return self.stepCover(after, 0)
    
    def merge(self, node, pos):
        if not node.parent:
            return 0
        
        if node.getValLen() >= self.mid_index:
            return 0
 
        parent = node.parent
        if pos:
            pre = parent.values[pos-1]
            bnode = parent.children[pos-1]
        else:
            pre = None
            bnode = parent.children[1]
 
        if bnode.getValLen() > self.mid_index:
            return self.rotate(node, bnode, parent, pre)
 
        if not pre:
            node.insert(parent.values.pop(0))
            bnode.children = node.children + bnode.children
        else:
            node.insert(parent.values.pop(pos-1))
            bnode.children = bnode.children + node.children
        bnode.values += node.values
        bnode.values.sort()         
        parent.children.remove(node)
        if parent.getValLen() == 0 and not parent.parent:
            self.root = bnode
            return 0
        
        if parent.getValLen() < self.mid_index:
            return self.merge(parent, parent.getPos())
            
    def rotate(self, node, bnode, parent, pre):
        if not pre:
            return self.leftRotate(node, bnode, parent)
        return self.rightRotate(node, bnode, parent)
    
    def leftRotate(self, node, bnode, parent):
        node.insert(parent.values.pop(0))
        parent.insert(bnode.values.pop(0))
        return 0
 
    def rightRotate(self, node, bnode, parent):
        pos = node.getPos()
        node.insert(parent.values.pop(pos-1))
        parent.insert(bnode.values.pop(-1))
        return 0
 
    def insert(self, *values):
        for value in values:
            node = self.find(value)
            length = node.insert(value)
            if length == self.scala:
                self._split(node)
 
    def delete(self, value):
        node = self.find(value)
        value_pos = node.values.index(value)
        node.values.remove(value)
        self.stepCover(node, value_pos)

def compare_word(node,index_list,word):

    for i in range(len(node.values)):
        
        wrange=index_list[node.values[i]]
        #print(len(node.children))
        if node.children==[]:
            #print('nc',i)
            if wrange[0]<=word and wrange[1]>=word:
            
                re=(None,node.values[i])
            
                return re
            elif i==len(node.values)-1:
                #print(node.values)
                re=(None,'notfound')
                return re
        elif wrange[0]<=word and wrange[1]>=word:
            
            #print(word)
            re=(None,node.values[i])
            
            return re
        
                
                
                
        elif word < wrange[0]:
            
            re=('Left',i)
            return re
            #elif word>= wrange[0] and word<=wrange[1]:
            
            #    re=(None,node.values[i])
            #    return re
        
        re=('Right',i+1)
    return re            
    
    
def find_id(btree,index_list,word,node: Node=None):
    
    if not node:
        
        _node=btree.root
    else:
        _node=node
        
    result=compare_word(_node,index_list,word)
    #print(result)
    #print(result,node.values)
    if result[0]==None:
        
        return result[1]
    
    else:
        #print(len(_node.children))
        #print(_node.values)
        return find_id(btree,index_list,word,_node.children[result[1]])
    
def KG_tree_construct(wordl):
     
    loc=0
    for w in wordl:
        
        if len(w[0])<1:
             
            loc=wordl.index(w)
            #print(loc)
            break
        elif w[0][0].upper()<'A' or w[0][0].upper()>'Z':
            
            loc=wordl.index(w)
            #print(loc)
            break
    
    AZ={}
    O={}
    for i in range(loc):
        
        AZ[i]=wordl[i]
        
    for i in range(loc,len(wordl)):
        O[i]=wordl[i]
    
    AZtree=BTree(scala=5)
    for i in range(len(AZ)):
        
        AZtree.insert(i)
        
    Otree=BTree(scala=5)
    for i in range(loc,len(wordl)):
        
        Otree.insert(i)
    
    return AZtree, Otree,AZ,O

def read_tree(tree_file):
    
    f=open('../KG/index KG/'+tree_file+'.data','rb')
    L=pickle.load(f)
    f.close()
    T=BTree(scala=5)
    for i in list(L.keys()):
        T.insert(i)
        
    return T,L

def open_file(tree,wl,entity):
    #print(entity)
    fid=find_id(tree,wl,entity,None)
    il=[fid]
    #print(il,entity)
    #print(type(fid))
    #print(il)
    
    if fid!='notfound' and fid!=max(wl.keys()):
        #print('+1')
        loc=fid+1
    else: 
        #print('=0')
        return il
    
    while wl[loc][0]==entity:
        
        il.append(loc)
        loc+=1
    return il
    
    
    
def find_triple(entity,isentity,tree,wl,needf,needp,indexfile):
    dbpedia_e=Namespace('http://dbpedia.org/resource/') 
    dbpedia_t=Namespace('http://dbpedia.org/ontology/')
    #if isentity==True:
    R=[]    
    
    if isentity==True:
        
        if needf=='ti':
            
            il=open_file(tree,wl,entity)
            if il==['notfound']:
                il=[]
            
            for ff in il:
                f=open(indexfile+'triple/'+str(ff)+'.data','rb')
                g=pickle.load(f)
                f.close()
            
                if needp=='p':
                    R+=g.predicates(dbpedia_e[entity],None)
                elif needp=='o':
                    R+=g.objects(dbpedia_e[entity],None)
                elif needp=='po':
                    R+=g.predicate_objects(dbpedia_e[entity])
                elif needp[0]=='op':
                    R+=g.predicates(dbpedia_e[entity],dbpedia_e[needp[1]])
                else:
                    R+=g.objects(dbpedia_e[entity],dbpedia_t[needp])
                    
        elif needf=='tr':
            #print(entity)
            il=open_file(tree,wl,entity)
            #print(il)
            if il==['notfound']:
                il=[]
            
            for ff in il:
                f=open(indexfile+'triple_reverse/'+str(ff)+'.data','rb')
                g=pickle.load(f)
                f.close()
            
                if needp=='p':
                    R+=g.predicates(None,dbpedia_e[entity])
                elif needp=='s':
                    R+=g.subjects(None,dbpedia_e[entity])
                #elif needp=='po':
                #    R+=g.predicate_objects(dbpedia_e[entity])            
            
        elif needf=='li':
            il=open_file(tree,wl,entity)
            if il==['notfound']:
                il=[]
            
            for ff in il:
                f=open(indexfile+'literal/'+str(ff)+'.data','rb')
                g=pickle.load(f)
                f.close()
            
                if needp=='p':
                    R+=g.predicates(dbpedia_e[entity],None)
                elif needp=='o':
                    R+=g.objects(dbpedia_e[entity],None)
                elif needp=='po':
                    R+=g.predicate_objects(dbpedia_e[entity])        
        
        elif needf=='ty':
            il=open_file(tree,wl,entity)
            if il==['notfound']:
                il=[]
            for ff in il:
                f=open(indexfile+'type/'+str(ff)+'.data','rb')
                g=pickle.load(f)
                f.close()
            
                if needp=='o':
                    R+=g.objects(dbpedia_e[entity],None)
                #elif needp=='po':
                #    R+=g.predicate_objects(dbpedia_e[entity])            
    else:
        il=open_file(tree,wl,entity)
        #print(il,'il',entity)
        if il==['notfound']:
            il=[]
        for ff in il:
            f=open(indexfile+'literal_reverse/'+str(ff)+'.data','rb')
            g=pickle.load(f)
            f.close()
        
            if needp=='p':
                R+=g.predicates(None,Literal(entity))
            elif needp=='s':
                R+=g.subjects(None,Literal(entity))            
    #print(il)        
    return R
        
#AZtree, Otree,AZ,O=KG_tree_construct(ti)  
#ii=find_id(AZtree,AZ,'Add',None)   

"""
f=open('../../KG/index KG/type/z_l.data','rb')
wl=pickle.load(f)
f.close()

AZtree, Otree,AZ,O=KG_tree_construct(wl)

f=open('../../KG/index KG/ty_wl_az.data','wb')
pickle.dump(AZ,f)
f.close()

f=open('../../KG/index KG/ty_wl_o.data','wb')
pickle.dump(O,f)
f.close()
"""  



    
    