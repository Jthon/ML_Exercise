import os
import cv2
import numpy as np
import time
import sys
import Linear_KNN as LK
import KD_Tree as KT
data_dir="./mnist/"
data_label_set=[]
data_image_set=[]
for sub_dir in os.listdir(data_dir):
    for filename in os.listdir(data_dir+sub_dir+"/"):
        if sub_dir!="test":
            data_label_set.append(int(sub_dir))
            image=cv2.imread(data_dir+sub_dir+"/"+filename,0)
            data_image_set.append(image)
data_label_set=np.array(data_label_set)
data_image_set=np.array(data_image_set)
data_image_set=np.array(data_image_set/255.0,np.float32)
data_image_set=np.reshape(data_image_set,(-1,784))

correct=0
linear=LK.LinearKnn(data_image_set,data_label_set,10)
start_time=time.time()
for filename in os.listdir("./mnist/test/"):
    test_label=filename.split(".")[0].split("_")[1]
    test_label=int(test_label)
    test_image=cv2.imread("./mnist/test/"+filename,0)
    test_image=test_image/255.0
    test_image=np.reshape(test_image,(784))
    prediction,_=linear.SearchKNN(test_image,5)
    if prediction==test_label:
        correct+=1
end_time=time.time()
print("MNIST TEST:")
print("Linear scan:correct rate=%f,running time=%f"%((correct/100.0),(end_time-start_time)))
correct=0
kd_tree_image_set=np.reshape(data_image_set,(-1,784))
kd_tree_label_set=data_label_set
kd_tree=KT.KD_Tree(kd_tree_image_set,kd_tree_label_set,10)
start_time=time.time()
for filename in os.listdir("./mnist/test/"):
    test_label=filename.split(".")[0].split("_")[1]
    test_label=int(test_label)
    test_image=cv2.imread("./mnist/test/"+filename,0)
    test_image=test_image/255.0
    test_image=np.reshape(test_image,(784))
    prediction,_=kd_tree.SearchNN(test_image,5)
    if prediction==test_label:
        correct+=1
end_time=time.time()
print("KD_tree:correct rate=%f,running time=%f"%((correct/100.0),(end_time-start_time)))


print("Random Data Test:")
input_data=np.random.rand(100000,5)*100
input_label=np.zeros((100000),np.int32)
search_data=np.random.rand(5)
tree=KT.KD_Tree(input_data,input_label,10)
linear=LK.LinearKnn(input_data,input_label,10)

time1=time.time()
result1,data1=tree.SearchNN(search_data,1)
time2=time.time()
result2,data2=linear.SearchKNN(search_data,1)
time3=time.time()

print("Linear scan:NN_Data={},running time=%f".format(data2)%(time3-time2))
print("KD Tree:NN_Data={},running time=%f".format(data1)%(time2-time1))