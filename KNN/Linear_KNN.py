import numpy as np
import Distance_helper as D_util
class LinearKnn:
    def __init__(self,data_x,data_y,kinds):
        self.input_x=data_x
        self.label_y=data_y
        self.kinds=kinds
    def SearchKNN(self,input_data,K):
        min_dis=[]
        min_member=[]
        min_data=[]
        for index,data in enumerate(self.input_x):
            distance=D_util.L2_Distance(data,input_data)
            if len(min_dis)<K:
                min_dis.append(distance)
                min_member.append(self.label_y[index])
                min_data.append(self.input_x[index])
            elif np.max(min_dis)>distance:
                max_index=np.argmax(min_dis)
                min_dis[max_index]=distance
                min_member[max_index]=self.label_y[index]
                min_data[max_index]=self.input_x[index]
        return self.Vote(min_member),min_data
    def Vote(self,result):
        nn_result=np.zeros((self.kinds),dtype=np.int32)
        for member in result:
            nn_result[member]=nn_result[member]+1
        return np.argmax(nn_result)



    