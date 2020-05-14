import numpy as np
import matplotlib.pyplot as plt
import copy

COLOR_PANEL=['#CD853F','#FF69B4','#00CED1','#00BFFF','#008B8B','#FF8C00','#2F4F4F','#FFFACD','#00FF00','#FFE4E1']
#Divide linear: y=5-x
group_onex=np.random.rand(100,1)
group_oney=5-5*group_onex+1*(0.5+np.random.rand(100,1))
group_one_feature=np.concatenate((group_onex,group_oney),axis=1)
group_one_label=np.ones(shape=group_onex.shape,dtype=np.float32)

group_twox=np.random.rand(100,1)
group_twoy=5-5*group_twox-1*(0.5+np.random.rand(100,1))
group_two_feature=np.concatenate((group_twox,group_twoy),axis=1)
group_two_label=-np.ones(shape=group_twox.shape,dtype=np.float32)

train_data=np.concatenate((group_one_feature,group_two_feature),axis=0)
train_label=np.concatenate((group_one_label,group_two_label),axis=0)

def Perceptron(w,b,training_data,training_label,lr_rate=1e-4,iteration=1000):
    loss_list=[]
    for i in range(0,iteration):
        pred_logits=np.matmul(training_data,w)+b
        
        judge=pred_logits*training_label
        index=np.where(judge<0)
        index=index[0]
        loss=-np.sum(np.clip(judge,a_max=0,a_min=None))
        grad_w=np.sum(training_data[index]*training_label[index],axis=0)
        grad_w=grad_w[:,np.newaxis]
        grad_b=np.sum(training_label[index],axis=0)
       
        w=w+lr_rate*grad_w
        b=b+lr_rate*grad_b
       
        loss_list.append(loss)
        
    return w,b,loss_list
init_w=np.random.rand(2,1)
init_b=np.random.rand(1)
w,b,loss_list1=Perceptron(init_w,init_b,train_data,train_label,1e-5,200)
w,b,loss_list2=Perceptron(init_w,init_b,train_data,train_label,1e-4,200)
w,b,loss_list3=Perceptron(init_w,init_b,train_data,train_label,1e-3,200)

x=np.linspace(0,1,500)
y=-(w[0]*x+b)/w[1]
fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.scatter(group_onex[:,0], group_oney[:,0],c=COLOR_PANEL[0])
ax.scatter(group_twox[:,0], group_twoy[:,0],c=COLOR_PANEL[1])
ax.plot(x,y,c=COLOR_PANEL[2])
bx = fig.add_subplot(1,2,2)
bx.plot(range(len(loss_list1)),loss_list1,c=COLOR_PANEL[0],label='lr_rate=1e-5')
bx.plot(range(len(loss_list2)),loss_list2,c=COLOR_PANEL[1],label='lr_rate=1e-4')
bx.plot(range(len(loss_list3)),loss_list3,c=COLOR_PANEL[2],label='lr_rate=1e-3')
plt.legend()
plt.show()

