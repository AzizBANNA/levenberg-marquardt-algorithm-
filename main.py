import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as m


def breit_wigner(x, a, b, c):
    #x=x.reshape(-1,1)
    gx= a/(((b-x)**2)+c)
    #print(type(gx))
    #gn=np.reshape(-1,1)
    return(np.reshape(gx,gx.shape[0]))
    #return gx

def eval_jacobian(x, func, params, h= 0.0001):
   arr=np.zeros((x.shape[0],len(params)))
   for i in range(0,len(params)):
      paramnew=params.copy()
      paramnew[i]=paramnew[i]+h
      arr[::,i]=(func(x,*paramnew)-func(x,*params))/h
      param_new=params
   return arr

def eval_errors(x_all , y_all, func, params ):
    return y_all - func(x_all , *params)

def get_params (jac,lma_lambda,e):
    ones_matrix=np.identity((jac.T.dot(jac)).shape[0])
    JTJ=jac.T.dot(jac)
    INV=np.linalg.inv(JTJ+ones_matrix*lma_lambda)
    return INV.dot(jac.T.dot(e))
    
def _lma_quality_measure (x , y , breit_wigner ,params , delta_params , jac , lma_lambda ):
    e=eval_errors ( x, y , breit_wigner ,params)
    e_next=eval_errors ( x, y , breit_wigner , params+delta_params )
    Help_calcul=lma_lambda*delta_params + jac.T.dot(e) 
    return ((e.dot(e) - e_next.dot(e_next))/((delta_params.T).dot(Help_calcul)))

def lma(x_all, y_all, func, param_guess, **kwargs):   
    params=param_guess
    lma_lambda=None
    param_change=1000
    while (param_change >= 0.00001):
        jac = eval_jacobian (x_all , func , params , h =0.0001)
        if lma_lambda is None :
            lma_lambda = np. linalg . norm ( jac.T.dot(jac))
        e = eval_errors (x_all , y_all , func , params)
        delta_params = get_params (jac,lma_lambda,e)
        
        lma_rho = _lma_quality_measure (x_all , y_all , func ,params , delta_params , jac , lma_lambda )
        if lma_rho > 0.75:
            lma_lambda /= 3
        elif lma_rho < 0.25:
            lma_lambda *= 2
        else :
            lma_lambda = lma_lambda
        if lma_rho > 0:
            params = [x_all+d for x_all, d in zip (params ,delta_params )]
        param_change = np.linalg.norm(delta_params)/np.linalg.norm(params)
    return params   

def lin2d(X, a, b):
    if X.shape==(X.shape[0],):
        return X
    else:
        return X[:, 0]*a+ X[:, 1]*b
        
        
if __name__=='__main__':
    df = pd.read_csv('breit_wigner.csv' ,  sep =',')
    x=df['x'].to_numpy()
    y=df['g'].to_numpy()
    print(breit_wigner(x, 1,2,3))
    print(eval_jacobian(x, breit_wigner ,[2,3,4], h=0.0001))
    fit = lma (lin2d(x, 1, 0), y, breit_wigner , np.random.rand(3)*1)
    x_axis = np.linspace(0,200,2000)
    fig=plt.figure()
    ax=fig.add_subplot()
    ax.plot(x_axis,breit_wigner(x_axis, *fit))
    ax.scatter(df['x'],df['g'])
    ax.set_title(f'Breit-Wigner function for params a={fit[0]}, b={fit[1]}, c={fit[2]}', fontsize=8)
    plt.savefig('plot.pdf')
