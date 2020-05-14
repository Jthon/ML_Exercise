import numpy as np
import KD_Node as Node
import QuickSort as mysort
import Distance_helper as D_util
import queue
import Linear_KNN as LK
import copy
import time
class KD_Tree:
    def __init__(self,input_x,label_y,kinds):
        self.data_x=input_x
        self.label_y=label_y
        self.visited_list=[]
        self.kinds=kinds
        self.feature_dim=input_x.shape[1]
        self.root=Node.Node()
        self.root.depth=0
        self.BuildTree(self.data_x,self.label_y,self.root)
        #self.PrintTree(self.root)
    def BuildTree(self,array_list,label_list,node):
        if len(array_list)==1:
            node.data=array_list[0]
            node.label=label_list[0]
        elif len(array_list)>1:
            keyword_index=node.depth%self.feature_dim
            left,middle,right,left_label,middle_label,right_label=mysort.MiddleArray(array_list,label_list,keyword_index)
            node.data=middle
            node.label=middle_label
            if len(left)>0:
                new_node=Node.Node()
                node.add_left(new_node)
                self.BuildTree(left,left_label,new_node)
            if len(right)>0:
                new_node=Node.Node()
                node.add_right(new_node)
                self.BuildTree(right,right_label,new_node)

    def Search_leaf(self,stack,input_data,node):
        if node!=None and id(node) not in self.visited_list:
            cut_dim=node.depth%self.feature_dim
            stack.put(node)
            if input_data[cut_dim]>node.data[cut_dim]:
                if node.right_child!=None:
                    node=node.right_child
                    self.Search_leaf(stack,input_data,node)
            else:
                if node.left_child!=None:
                    node=node.left_child
                    self.Search_leaf(stack,input_data,node)
                
    def Vote(self,label_result):
        result=np.zeros((self.kinds))
        for member in label_result:
            result[member]+=1
        return np.argmax(result)
    def SearchNN(self,input_data,k):
        self.visited_list=[]
        min_distance=[]
        min_data=[]
        min_member=[]
        stack=queue.LifoQueue()
        self.Search_leaf(stack,input_data,self.root)
        while stack.empty()==False:
            node=stack.get()
            distance=D_util.L2_Distance(node.data,input_data)
            self.visited_list.append(id(node))
            if len(min_distance)<k:
                min_distance.append(distance)
                min_data.append(node.data)
                min_member.append(node.label)
            elif np.max(min_distance)>distance:
                max_index=np.argmax(min_distance)
                min_distance[max_index]=distance
                min_data[max_index]=node.data
                min_member[max_index]=node.label
            if node.father==None:
                break
            cut_dim=node.father.depth%self.feature_dim
            if np.abs(node.father.data[cut_dim]-input_data[cut_dim])<np.max(min_distance):
                if node.left_or_right==0:
                    self.Search_leaf(stack,input_data,node.father.right_child)
                elif node.left_or_right==1:
                    self.Search_leaf(stack,input_data,node.father.left_child)
        return self.Vote(min_member),min_data
    def PrintTree(self,node):
        print("data={},depth=%d,left_or_right={}".format(node.data,node.left_or_right)%node.depth)
        if node.left_child!=None:
            self.PrintTree(node.left_child)
        if node.right_child!=None:
            self.PrintTree(node.right_child)










