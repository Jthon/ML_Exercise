import cv2
import os
import numpy as np
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
data_image_set=np.reshape(data_image_set,(-1,784))
def Build_Bayes(data_set,label_set,species,dims,values):
    prior_dis=np.zeros((species),dtype=np.int32)
    for i in range(0,species):
        prior_dis[i]=np.where(label_set==0)[0].shape[0]
    model_mat=np.ones(shape=(species,dims,values),dtype=np.int32)
    for i in range(0,data_set.shape[0]):
        label=label_set[i]
        for j in range(0,dims):
            pixel_value=data_set[i][j]
            model_mat[label][j][pixel_value]+=1
    prior_dis=prior_dis[:,np.newaxis,np.newaxis]
    prior_dis=prior_dis+values
    model_mat=model_mat/prior_dis
    return model_mat
def Bayesian_Predict(Bayes_Matrix,input_image):
    result=1
    for i in range(0,input_image.shape[0]):
        pixel_value=input_image[i]
        result=result*Bayes_Matrix[:,i,pixel_value]*15
    return np.argmax(result)
model_mat=Build_Bayes(data_image_set,data_label_set,10,784,256)
correct=0
for filename in os.listdir("./mnist/test/"):
    test_label=filename.split(".")[0].split("_")[1]
    test_label=int(test_label)
    test_image=cv2.imread("./mnist/test/"+filename,0)
    test_image=np.reshape(test_image,(784))
    prediction=Bayesian_Predict(model_mat,test_image)
    if prediction==test_label:
        correct+=1
print("correct rate=%f"%(correct/100.0))


