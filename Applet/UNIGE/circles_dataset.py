# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:46:00 2022

@author: giuli
"""
import numbers
from sklearn.utils import check_random_state
from sklearn.utils import shuffle as util_shuffle
import numpy as np

def innested_circles(
    n_samples=100, *, shuffle=True, noise=None, random_state=None, num_class, dist=0
):
    factor = ((dist-0)*(0-2)/(2-0))+2
    if isinstance(n_samples, numbers.Integral):
        if(num_class==2):
            n_samples_out = n_samples // 2
            n_samples_in = n_samples - n_samples_out
        if(num_class==3):
            n_samples_out = n_samples // 3
            n_samples_in = (n_samples - n_samples_out)//2
            n_samples_3 = n_samples - n_samples_out - n_samples_in
        if(num_class==4):
            n_samples_out = n_samples // 4
            n_samples_in = (n_samples - n_samples_out)//3
            n_samples_3 = (n_samples - n_samples_out - n_samples_in)//2
            n_samples_4 = n_samples - n_samples_out - n_samples_in - n_samples_3
        if(num_class==5):
            n_samples_out = n_samples // 5
            n_samples_in = (n_samples - n_samples_out)//4
            n_samples_3 = (n_samples - n_samples_out - n_samples_in)//3
            n_samples_4 = (n_samples - n_samples_out - n_samples_in - n_samples_3)//2
            n_samples_5 = n_samples - n_samples_out - n_samples_in - n_samples_3 - n_samples_4 
    else:
        try:
            n_samples_out, n_samples_in = n_samples
        except ValueError as e:
            raise ValueError(
                "`n_samples` can be either an int or a two-element tuple."
            ) from e

    generator = check_random_state(random_state)
    # so as not to have the first point = last point, we set endpoint=False
    if(num_class==3):
        linspace_out = np.linspace(0, 2 * np.pi, n_samples_out, endpoint=False)
        linspace_in = np.linspace(0, 2 * np.pi, n_samples_in, endpoint=False)
        outer_circ_x = np.cos(linspace_out)*2
        outer_circ_y = np.sin(linspace_out)*2
        inner_circ_x = np.cos(linspace_in)*(0+factor)
        inner_circ_y = np.sin(linspace_in)*(0+factor)
        linspace = np.linspace(0, 2 * np.pi, n_samples_3, endpoint=False)
        circ_x = np.cos(linspace)*(4-factor)
        circ_y = np.sin(linspace)*(4-factor)
        X= np.concatenate((np.concatenate((outer_circ_x.reshape(-1,1), circ_x.reshape(-1,1), inner_circ_x.reshape(-1,1)), axis=0),np.concatenate((outer_circ_y.reshape(-1,1), circ_y.reshape(-1,1), inner_circ_y.reshape(-1,1)), axis=0)),axis=1)
        y = np.hstack([np.zeros(n_samples_out, dtype=np.intp), np.ones(n_samples_in, dtype=np.intp), np.ones(n_samples_3, dtype=np.intp)*2])    
    if(num_class==4):
        linspace_out = np.linspace(0, 2 * np.pi, n_samples_out, endpoint=False)
        linspace_in = np.linspace(0, 2 * np.pi, n_samples_in, endpoint=False)
        linspace_3 = np.linspace(0, 2 * np.pi, n_samples_3, endpoint=False)
        linspace_4 = np.linspace(0, 2 * np.pi, n_samples_4, endpoint=False)
        if(factor==0):           
            outer_circ_x = np.cos(linspace_out)*(2)
            outer_circ_y = np.sin(linspace_out)*(2)
            inner_circ_x = np.cos(linspace_in)*0
            inner_circ_y = np.sin(linspace_in)*0
            circ_x = np.cos(linspace_3)*4
            circ_y = np.sin(linspace_3)*4 
            circ_x_4 = np.cos(linspace_4)*(6)
            circ_y_4 = np.sin(linspace_4)*(6)

        else:  
            outer_circ_x = np.cos(linspace_out)*(2)
            outer_circ_y = np.sin(linspace_out)*(2)
            inner_circ_x = np.cos(linspace_in)*(0+factor)
            inner_circ_y = np.sin(linspace_in)*(0+factor)
            circ_x = np.cos(linspace_3)*(4-factor)
            circ_y = np.sin(linspace_3)*(4-factor) 
            circ_x_4 = np.cos(linspace_4)*(6-np.power(2,factor))
            circ_y_4 = np.sin(linspace_4)*(6-np.power(2,factor))
        X= np.concatenate((np.concatenate((circ_x_4.reshape(-1,1), outer_circ_x.reshape(-1,1), circ_x.reshape(-1,1), inner_circ_x.reshape(-1,1)), axis=0),\
                           np.concatenate((circ_y_4.reshape(-1,1),outer_circ_y.reshape(-1,1), circ_y.reshape(-1,1), inner_circ_y.reshape(-1,1)), axis=0)),axis=1)
        y = np.hstack([np.zeros(n_samples_out, dtype=np.intp), np.ones(n_samples_in, dtype=np.intp), np.ones(n_samples_3, dtype=np.intp)*2, np.ones(n_samples_3, dtype=np.intp)*3]) 
    if(num_class==5):
        linspace_out = np.linspace(0, 2 * np.pi, n_samples_out, endpoint=False)
        linspace_in = np.linspace(0, 2 * np.pi, n_samples_in, endpoint=False)
        linspace_3 = np.linspace(0, 2 * np.pi, n_samples_3, endpoint=False)
        linspace_4 = np.linspace(0, 2 * np.pi, n_samples_4, endpoint=False)
        linspace_5 = np.linspace(0, 2 * np.pi, n_samples_5, endpoint=False)
        if(factor==0):           
            outer_circ_x = np.cos(linspace_out)*(2)
            outer_circ_y = np.sin(linspace_out)*(2)
            inner_circ_x = np.cos(linspace_in)*(0)
            inner_circ_y = np.sin(linspace_in)*(0)
            circ_x = np.cos(linspace_3)*(4)
            circ_y = np.sin(linspace_3)*(4)
            circ_x_4 = np.cos(linspace_4)*(6)
            circ_y_4 = np.sin(linspace_4)*(6) 
            circ_x_5 = np.cos(linspace_5)*(8)
            circ_y_5 = np.sin(linspace_5)*(8)
        else:
            outer_circ_x = np.cos(linspace_out)*(2+factor)
            outer_circ_y = np.sin(linspace_out)*(2+factor)
            inner_circ_x = np.cos(linspace_in)*(0+np.power(2,factor))
            inner_circ_y = np.sin(linspace_in)*(0+np.power(2,factor))
            circ_x = np.cos(linspace_3)*(4)
            circ_y = np.sin(linspace_3)*(4)
            circ_x_4 = np.cos(linspace_4)*(6-factor)
            circ_y_4 = np.sin(linspace_4)*(6-factor) 
            circ_x_5 = np.cos(linspace_5)*(8-np.power(2,factor))
            circ_y_5 = np.sin(linspace_5)*(8-np.power(2,factor))
        X= np.concatenate((np.concatenate((circ_x_5.reshape(-1,1),circ_x_4.reshape(-1,1),outer_circ_x.reshape(-1,1), circ_x.reshape(-1,1), inner_circ_x.reshape(-1,1)), axis=0),\
                           np.concatenate((circ_y_5.reshape(-1,1),circ_y_4.reshape(-1,1),outer_circ_y.reshape(-1,1), circ_y.reshape(-1,1), inner_circ_y.reshape(-1,1)), axis=0)),axis=1)
        y = np.hstack([np.zeros(n_samples_out, dtype=np.intp), np.ones(n_samples_in, dtype=np.intp), np.ones(n_samples_3, dtype=np.intp)*2, np.ones(n_samples_3, dtype=np.intp)*3, np.ones(n_samples_3, dtype=np.intp)*4]) 
    if(num_class==2):
        linspace_out = np.linspace(0, 2 * np.pi, n_samples_out, endpoint=False)
        linspace_in = np.linspace(0, 2 * np.pi, n_samples_in, endpoint=False)
        outer_circ_x = np.cos(linspace_out)*(2)
        outer_circ_y = np.sin(linspace_out)*(2)
        inner_circ_x = np.cos(linspace_in)*(0+factor)
        inner_circ_y = np.sin(linspace_in)*(0+factor)
        X = np.vstack([np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]).T
        y = np.hstack([np.zeros(n_samples_out, dtype=np.intp), np.ones(n_samples_in, dtype=np.intp)])
    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if noise is not None:
        X += generator.normal(scale=noise, size=X.shape)

    return X, y