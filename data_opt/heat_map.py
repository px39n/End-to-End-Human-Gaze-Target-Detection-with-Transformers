import numpy as np
import torch
from scipy import integrate

def gaussion_quad(i,j,mu_x=0.0,mu_y=0.0,stride=1):
    def gaussian(x, mu=0, sigma=1):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    def bivariate_gaussian(x,y):
        # x=np.sqrt((x**2+y**2))
        return gaussian(x)*gaussian(y)

    # x=distance.chebyshev((i,j), (h-1,w-1))
    a,b=stride*i-mu_x,stride*j-mu_y
    return integrate.dblquad(bivariate_gaussian,a,a+stride,b,b+stride)[0]

class HeatMap():
    def __init__(self,w=5,h=5,stride_dim=50):
        self.h=h
        self.w=w
        self.stride=stride_dim//h

    def init_heat_map(self,x,y):
        heatMap=[[gaussion_quad(i,j,x,y,self.stride) for i in range(self.w)] for j in range(self.h)]
        return torch.tensor(heatMap)

    def gaze_index(self,im_h,im_w,g_x,g_y):
        index_x=self.h*g_x/im_h
        index_y=self.w*g_y/im_w

        x,y=index_x*self.stride,index_y*self.stride
        return x,y

    def __call__(self, im_w,im_h,g_x,g_y,*args, **kwargs):

        index_x,index_y=self.gaze_index(im_w,im_h,g_x,g_y)
        return self.init_heat_map(index_x,index_y)



