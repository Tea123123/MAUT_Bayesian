import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import math

import matplotlib.ticker as plticker
from datetime import datetime
import sys

#attribute-specific mean, uncertainty, distribution
class Attribute:
    def __init__(self, m, std, a, ea, dist="normal"):
        self.mean = m
        self.std = std
        self.a = a #risk attitude
        self.ea = ea #estimation uncertainty
        
        if dist == "normal":
            self.value = np.random.normal(m, std)
        elif dist == "lognormal":
            self.value = np.random.lognormal(m, std)     
        
        self.U_f = U_f(self.a, self.std*(-3), self.std*3) #range set to include most of the randomly generated values
        
        self.ux = self.U_f.value(self.value)
        
        self.y = np.random.normal(self.value, self.ea)
        self.uy = self.U_f.value(self.y)
        
        s2 = self.std**2
        t2 = self.ea**2        
        m = (t2/(s2 + t2))*self.mean + (s2/(s2 + t2))*self.y
        std = np.sqrt((s2*t2/(s2 + t2)))
        
        #for right way 
        uz_l = []
        for k in range(1000):
            uz = self.U_f.value(np.random.normal(m, std))
            uz_l.append(uz)
        
        self.uz = np.mean(uz_l)
        
        #for wrong way 
        mean = np.mean(np.random.normal(m, std, 1000))
        self.uz2 = self.U_f.value(mean)
        
    def add_u(self, U): #add the attribute-specific utility function
        self.U_f = U
        self.a = U.a
    
    def describe(self, graph=False):
        print("Mean: ", self.mean)
        print("Std: ", self.std)
        print("Risk parameter a: ", self.a)
        print("Estimation uncertainty: ", self.ea)
        print("")
        print("Value: %.3f" %(self.value))
        print("ux: %.3f, uy: %.3f, uz: %.3f, uz3: %.2f" %(self.ux, self.uy, self.uz, self.uz2))
        
        if graph:
            print("Utility curve: ")
            self.U_f.graph()
            
 #attribute-specific utility function
class U_f:
    def __init__(self, a, x_min, x_max): #parameter: performance x, risk param a \in [-1,1],
        self.a = a
        #self.std = s 
        #self.x =x
        self.min = x_min
        self.max = x_max
        self.mean = (x_min+x_max)/2.0
        
    def value(self, x):
        if x <= self.min:
            return 0
        elif x >= self.max:
            return 1
        elif self.a == 0:
            return (1/(self.max-self.min))*(x-self.mean) + 0.5
        else:
            def b(z):
                return 1-math.e**(-self.a*z)
            
            if self.a >0:
                self.y_up = -b(self.min-self.mean)
            else:
                self.y_up = -b(self.mean-self.max)
                
            up = b(x-self.mean) + self.y_up
            down = (b(self.max-self.mean)-b(self.min-self.mean))  
            return up/down
        
    def graph(self):
        X = np.arange(self.min, self.max, 0.05)
        Y1 = [U_f(0.367, self.min, self.max).value(x) for x in X]
        Y2 = [U_f(-0.367, self.min, self.max).value(x) for x in X]
        Y3 = [U_f(0,self.min, self.max).value(x) for x in X] 
        Y4 = [U_f(self.a, self.min, self.max).value(x) for x in X]
        plt.plot(X,Y3, label=r'$a_i=0$', color='black') 
        plt.plot(X,Y1, label=r'$a_i=0.367$', color='black', linestyle='dotted') 
        plt.plot(X,Y2, label=r'$a_i=-0.367$', color='black', linestyle='dashed') 
        plt.plot(X,Y4, label=r'$a_i=$' + str(self.a), color='red') 
        #plt.axvline(0, color='k')
        #plt.axhline(0, color='k')
        plt.xlabel(r"$x_{ji}$")
        plt.ylabel(r"$u_i(x_{ji})$")

        plt.yticks(np.arange(0, 1.25, 0.25))

        plt.grid(which='major', axis='both', linestyle='-')
        plt.legend()
        plt.show()
        
class Alternative:
    def __init__(self, n=3):#number of attribute
        self.n = n
        self.w = [1/n]*n
        self.attr_l = [Attribute(m=0, std=1, a=0, ea=1) for i in range(n)] #original attribute
        
    def re_cal(self):
        self.x = []
        self.y = []
        self.z = []
        self.z2 = []
        
        for attr in self.attr_l:
            self.x.append(attr.ux) 
            self.y.append(attr.uy)
            self.z.append(attr.uz)
            self.z2.append(attr.uz2)
            
        self.x_val = np.dot(self.w, self.x)
        self.y_val = np.dot(self.w, self.y)
        self.z_val = np.dot(self.w, self.z)
        self.z2_val = np.dot(self.w, self.z2)
        
    def add_attributes(self, attr_l):
        self.attr_l = attr_l
        self.re_cal()
        
    def set_weight(self, w):
        self.w = w
        self.re_cal()
        
    def describe(self):
        self.re_cal()
        print("Number of attributes: ", self.n)
        def round_list(l):
            return [round(x,3) for x in l]
        print("Attribute weights: ", round_list(self.w))
        print("x: ", round_list(self.x))
        print("y: ", round_list(self.y))
        print("z: ", round_list(self.z))
        print("z2: ", round_list(self.z2))
        print("Ux: %.3f, Uy: %.3f, Uz: %.3f, Uz2: %.3f" % (self.x_val, self.y_val, self.z_val, self.z2_val))
        
        
class Problem:
    def __init__(self, m=5, n=3): 
        self.m = m #number of alternatives
        self.n = n #number of attributes
        
        self.Alt_l = [Alternative(n=self.n) for i in range(self.m)]
        
    def re_cal(self):
        self.x_l = [alt.x_val for alt in self.Alt_l]
        self.y_l = [alt.y_val for alt in self.Alt_l]  
        self.z_l = [alt.z_val for alt in self.Alt_l]
        self.z2_l = [alt.z2_val for alt in self.Alt_l]
        
        self.max_x = max(self.x_l)
        self.max_y = max(self.y_l)
        self.max_z = max(self.z_l)
        self.max_z2 = max(self.z2_l)
        
        self.max_x_i = np.argmax(self.x_l)
        self.max_y_i = np.argmax(self.y_l)   
        self.max_z_i = np.argmax(self.z_l)
        self.max_z2_i = np.argmax(self.z2_l)
        
        self.pds_y = (self.x_l[self.max_y_i] - self.max_y)/self.max_y
        self.pds_z = (self.x_l[self.max_z_i] - self.max_z)/self.max_z
        self.pds_z2 = (self.x_l[self.max_z2_i] - self.max_z2)/self.max_z2
        
        self.eu_yz = self.x_l[self.max_y_i] - self.x_l[self.max_z_i]
        self.eu_yz2 = self.x_l[self.max_y_i] - self.x_l[self.max_z2_i]
        self.eu_zz2 = self.x_l[self.max_z2_i] - self.x_l[self.max_z_i]
        
    def set_parameters(self, w, a, mean, std, ea): #list for parameters for each attribute
        attr_l = []
        
        for j in range(self.m):
            print("Alternative ", str(j))
            print("")
            Alt = Alternative(self.n)
                 
            print("Attribute settings:")
            attr_l = []
            for i in range(self.n): #generate attributes
                attr = Attribute(m=mean[i], std=std[i], a=a[i], ea=ea[i], dist="normal")
                print("Attribute ", str(i))
                print(attr.describe())
                attr_l.append(attr)
                
            Alt.add_attributes(attr_l)
            Alt.set_weight(w)
            self.Alt_l[j] = Alt
            Alt.describe()
            print("")
        
        self.re_cal()
        
    def describe(self):
        self.re_cal()
        print("Number of alternatives: ", self.m)
        print("Number of attributes: ", self.n)
        for j in range(self.m):
            print("Alternative ", str(j+1))
            self.Alt_l[j].describe()
            print("")                 
            print("Attribute settings: ")
            for i in range(self.n): 
                print("Attribute ", str(i+1))
                print(self.Alt_l[j].attr_l[i].describe())                       
            print("") 
                
        def round_list(l):
            return [round(x,3) for x in l]       
        print("X of alternatives: ", round_list(self.x_l))
        print("Y of alternatives: ", round_list(self.y_l))
        print("Z of alternatives: ", round_list(self.z_l))
        print("Z2 of alternatives: ", round_list(self.z2_l))
        print("")
        print("Relative PDS of Y: ", self.pds_y)
        print("Relative PDS of Z: ", self.pds_z)
        print("Relative PDS of z2: ", self.pds_z2)
        print("")
        print("EU YZ: ", self.eu_yz)
        print("EU YZ2: ", self.eu_yz2)
        print("EU ZZ2: ", self.eu_zz2)
        
        
        
class Simulation:
    def __init__(self,  n_alt=5, n_attr=3, w=[1/3, 1/3, 1/3], a= [0,0,0], mean= [0,0,0], std=[1,1,1], ea=[1,1,1]):
        self.pds_y = []
        self.pds_z = []
        self.pds_z2 = []
        
        self.eu_yz = []
        self.eu_yz2 = []
        self.eu_zz2 = []
        
        self.toc_y = []
        self.toc_z = []
        self.toc_z2 = []
        
        self.m = n_alt
        self.n = n_attr
        
        self.w = w
        self.a = a
        self.mean= mean
        self.std = std
        self.ea = ea
        
        self.p = Problem(m=self.m, n=self.n)
        self.p.set_parameters(w=self.w, a=self.a, mean=self.mean, std=self.std, ea=self.ea)
        self.p.describe()
        
    def run(self, rep=10000):
        
        start_time = datetime.now()
       
        for i in range(rep):
    
            p = Problem(m=self.m, n=self.n)
            p.set_parameters(w=self.w, a=self.a, mean=self.mean, std=self.std, ea=self.ea)

            self.pds_y.append(p.pds_y)
            self.pds_z.append(p.pds_z)
            self.pds_z2.append(p.pds_z2)
            
            self.eu_yz.append(p.eu_yz)
            self.eu_yz2.append(p.eu_yz2)
            self.eu_zz2.append(p.eu_zz2)
            
            self.toc_y.append(p.max_x_i == p.max_y_i)
            self.toc_z.append(p.max_x_i == p.max_z_i)
            self.toc_z2.append(p.max_x_i == p.max_z2_i)
            
        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))
        
    def save_output(self, filename):
        d = {}
        d["pds_y"] = self.pds_y
        d["pds_z"] = self.pds_z
        d["pds_z2"] = self.pds_z2
        d["eu_yz"] = self.eu_yz
        d["eu_yz2"] = self.eu_yz2
        d["eu_zz2"] = self.eu_zz2
        d["toc_y"] = self.toc_y
        d["toc_z"] = self.toc_z
        d["toc_z2"] = self.toc_z2
        
        df = pd.DataFrame(d)
        df.to_csv(filename + ".csv")

        orig_stdout = sys.stdout
        f = open(filename + '.txt', 'w')
        sys.stdout = f
        
        self.p.describe()

        sys.stdout = orig_stdout
        f.close()
        