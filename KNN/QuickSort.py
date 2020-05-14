import numpy as np
import copy
def AdjustOrder(array,label_array,keyword_index,left,right):
    i=left
    j=right
    origin=copy.deepcopy(array[i])
    base=array[i][keyword_index]
    label_base=copy.deepcopy(label_array[i])
    while i<j:
        while array[j][keyword_index]>=base:
            j=j-1
            if j==i:
                break
        array[i]=array[j]
        label_array[i]=label_array[j]
        while array[i][keyword_index]<base:
            i=i+1
            if j==i:
                break
        array[j]=array[i]
        label_array[j]=label_array[i]
    array[i]=origin
    label_array[i]=label_base
    return i
def QuickSort(array,label_array,keyword_index,left,right):
    if left<right:
        mid=AdjustOrder(array,label_array,keyword_index,left,right)
        QuickSort(array,label_array,keyword_index,left,mid)
        QuickSort(array,label_array,keyword_index,mid+1,right)
def MiddleArray(array,label_array,keyword_index):
    QuickSort(array,label_array,keyword_index,0,array.shape[0]-1)
    middle=array[int(array.shape[0]/2)]
    left=array[0:int(array.shape[0]/2)]
    right=array[int(array.shape[0]/2)+1:array.shape[0]]
    left_label=label_array[0:int(array.shape[0]/2)]
    middle_label=label_array[int(array.shape[0]/2)]
    right_label=label_array[int(array.shape[0]/2)+1:array.shape[0]]
    return left,middle,right,left_label,middle_label,right_label


