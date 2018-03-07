import numpy as np
import random

class SVMClassifier():
    def __init__(self,max_itre = 200,kernel_type = 'linear',kernel_parameter = 1,C = 1.0,epsilon = 0.001):
        self.max_itre = max_itre
        self.kernels = {
            'linear':self.kernel_linear,
            'quadratic':self.kernel_quadratic,
            'gauss':self.kernel_guass
        }
        self.kernel_parameter = 1
        self.kernel = self.kernels[kernel_type]
        self.C = C
        self.epsilon = epsilon
        self.b = 0
        self.alpha = 0
        self.y = 0
        self.x = 0
        self.label1 = 0
        self.label2 = 0
        self.dic = {}
    
    def fit(self,X,y):
        self.x = X
        n,d = self.x.shape
        self.y = self.preprocess(y)
        self.alpha = np.zeros((1,n))
        count = 0
        examineAll = 1
        loop = True
        while loop:
            count += 1
            times1 = 0
            times2 = 0
            alpha_prev = self.alpha.copy()
            for i in range(n):
                if examineAll == -1:
                    if 0< self.alpha[0,i] < self.C and abs(self.cal_err(i)) > self.epsilon:
                        index_1 = i
                        index_2 = self.get_j(index_1)
                        alpha_1_new,alpha_2_new,self.b = self.update(index_1,index_2)
                        self.alpha[0,index_1] = alpha_1_new
                        self.alpha[0,index_2] = alpha_2_new
                    else:times1 += 1
                if examineAll == 1:
                    EE_i = self.cal_err(i)
                    if (0<self.alpha[0,i]<self.C and abs(EE_i) > self.epsilon) or (self.alpha[0,i] == 0 and EE_i < self.epsilon) or (self.alpha[0,i] == self.C and EE_i > -self.epsilon):
                        index_1 = i
                        index_2 = self.get_j(index_1)
                        alpha_1_new,alpha_2_new,self.b = self.update(index_1,index_2)
                        self.alpha[0,index_1] = alpha_1_new
                        self.alpha[0,index_2] = alpha_2_new
                    else:times2 += 1
            summ = np.linalg.norm((self.alpha-alpha_prev))
            print(summ)
            if summ < self.epsilon:
                break
            if times1 > times2:
                examineAll = 1
            else:
                examineAll = -1
            if count > self.max_itre:
                print("algorithm doesn't convergent")
                break
    
    def predict(self,x):
        prelabel = np.zeros((x.shape[0],1))
        for i in range(x.shape[0]):
            value = self.cal_g(x[i,:])
            prelabel[i,0] = self.dic[self.judge(value)]
        return prelabel
                  
    def preprocess(self,y):
        a = max(y)
        self.label1 = a
        self.label2 = min(y)
        self.dic = {1:self.label1,-1:self.label2}
        for i in range(y.shape[0]):
            if y[i,0] == a:
                y[i,0] = 1
            else:
                y[i,0]  = -1
        return y
                        
    def get_H_L(self,alpha_2_new_prime,index_1,index_2):
        if self.y[index_1,0] != self.y[index_2,0]:
            L = max(0,self.alpha[0,index_2]-self.alpha[0,index_1])
            H = min(self.C,self.C+self.alpha[0,index_2]-self.alpha[0,index_1])
        else:
            L = max(0,self.alpha[0,index_2]+self.alpha[0,index_1]-self.C)
            H = min(self.C,self.alpha[0,index_2]+self.alpha[0,index_1])
        return H,L             
                        
    def update(self,index_1,index_2):
        E_1 = self.cal_E(index_1)     
        E_2 = self.cal_E(index_2) 
        K_11 = self.kernel(self.x[index_1,:],self.x[index_1,:])
        K_22 = self.kernel(self.x[index_2,:],self.x[index_2,:])       
        K_12 = self.kernel(self.x[index_1,:],self.x[index_2,:]) 
        K_21 = self.kernel(self.x[index_2,:],self.x[index_1,:]) 
        n = K_11+K_22-K_12
        alpha_2_new_prime = self.alpha[0,index_2] + self.y[index_2,0]*(E_1-E_2)/n
        H,L = self.get_H_L(alpha_2_new_prime,index_1,index_2) 
        if alpha_2_new_prime > H:
            alpha_2_new = H
        elif alpha_2_new_prime < L:
            alpha_2_new = L
        else: alpha_2_new = alpha_2_new_prime
        alpha_1_new = self.alpha[0,index_1] + self.y[index_1,0]*self.y[index_2,0]*(self.alpha[0,index_2]-alpha_2_new)
 
        b_new = -E_1 - self.y[index_1,0]*K_11*(alpha_1_new - self.alpha[0,index_1]) - self.y[index_2,0]*K_21*(alpha_2_new - self.alpha[0,index_2]) + self.b
        if not (0<alpha_1_new<self.C and 0<alpha_2_new<self.C):
            b_new2 = -E_2 - self.y[index_1,0]*K_12*(alpha_1_new - self.alpha[0,index_1]) - self.y[index_2,0]*K_22*(alpha_2_new - self.alpha[0,index_2]) + self.b
            b_new = (b_new+b_new2)/2.        
        return alpha_1_new,alpha_2_new,b_new
             
    
    def get_Elist(self):
        Elist = []
        for i in range(self.x.shape[0]):
            Elist.append(self.cal_E(i))
        return Elist
    
    def get_j(self,index_1):
        E_1 = self.cal_E(index_1)
        Elist = self.get_Elist()
        while True:
            if E_1 > 0:
                lis = np.where(np.mat(Elist) < 0)[1]
                j = lis[random.randint(0,len(lis)-1)]
            else: 
                lis = np.where(np.mat(Elist) > 0)[1]
                j = lis[random.randint(0,len(lis)-1)]
            if j == index_1:
                Elist.remove(Elist[j])
            else:       
                break
        return j
    
    def cal_err(self,i):
        return self.y[i,0]*self.cal_g(self.x[i,:]) - 1
    
    def cal_E(self,i):
        return self.cal_g(self.x[i,:]) - self.y[i,0]
    
    def judge(self,gx):
        return np.sign(gx).astype(int)
        
    def cal_g(self,x):
        return float(np.multiply(self.alpha,self.y.T).dot(self.kernel(x,self.x).T) + self.b)    
                           
    def kernel_linear(self,x1,x2):
        return np.dot(x1,x2.T)    
        
    def kernel_guass(self,x1,x2):
        x1 = np.mat(x1)
        x2 = np.mat(x2)
        return np.exp(-1*(np.mean(np.multiply((x1-x2),(x1-x2)),axis = 1))/(self.kernel_parameter**2)).T
            
    def kernel_quadratic(self,x1,x2):
        return np.multiply(np.dot(x1,x2.T),np.dot(x1,x2.T))
  
