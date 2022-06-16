from scipy.optimize import minimize
import numpy as np
from scipy.special import gamma


class DCC:
    
    def __init__(self, dist = 'norm'):
        if dist == 'norm' or dist == 't':
            self.dist = dist
        else: 
            print("Takes pdf name as param: 'norm' or 't'.")
            
    def garch_fit(self, returns):
        res = minimize( self.garch_loglike, (0.01, 0.01, 0.94), args = returns,
              bounds = ((1e-6, 1), (1e-6, 1), (1e-6, 1)))
        return res.x

    def garch_loglike(self, params, returns):
        T = len(returns)
        var_t = self.garch_var(params, returns)
        LogL = np.sum(-np.log(2*np.pi*var_t)) - np.sum( (returns.A1**2)/(2*var_t))
        return -LogL

    def garch_var(self, params, returns):
        T = len(returns)
        omega = params[0]
        alpha = params[1]
        beta = params[2]
        var_t = np.zeros(T)     
        for i in range(T):
            if i==0:
                var_t[i] = returns[i]**2
            else: 
                var_t[i] = omega + alpha*(returns[i-1]**2) + beta*var_t[i-1]
        return var_t        
        
    def mgarch_loglike(self, params, D_t):
        # No of assets
        a = params[0]
        b = params[1]
        Q_bar = np.cov(self.rt.reshape(self.N, self.T))

        Q_t = np.zeros((self.T,self.N,self.N))
        R_t = np.zeros((self.T,self.N,self.N))
        H_t = np.zeros((self.T,self.N,self.N))
        
        Q_t[0] = np.matmul(self.rt[0].T/2, self.rt[0]/2)

        loglike = 0
        for i in range(1,self.T):
            dts = np.diag(D_t[i])
            dtinv = np.linalg.inv(dts)
            et = dtinv*self.rt[i].T
            Q_t[i] = (1-a-b)*Q_bar + a*(et*et.T) + b*Q_t[i-1]
            qts = np.linalg.inv(np.sqrt(np.diag(np.diag(Q_t[i]))))

            R_t[i] = np.matmul(qts, np.matmul(Q_t[i], qts))


            H_t[i] = np.matmul(dts, np.matmul(R_t[i], dts))   

            loglike = loglike + self.N*np.log(2*np.pi) + \
                      2*np.log(D_t[i]).sum() + \
                      np.log(np.linalg.det(R_t[i])) + \
                      np.matmul(self.rt[i], (np.matmul( np.linalg.inv(H_t[i]), self.rt[i].T)))


        return loglike

    
    def mgarch_logliket(self, params, D_t):
        # No of assets
        a = params[0]
        b = params[1]
        dof = params[2]
        Q_bar = np.cov(self.rt.reshape(self.N, self.T))

        Q_t = np.zeros((self.T,self.N,self.N))
        R_t = np.zeros((self.T,self.N,self.N))
        H_t = np.zeros((self.T,self.N,self.N))
        
        Q_t[0] = np.matmul(self.rt[0].T/2, self.rt[0]/2)

        loglike = 0
        for i in range(1,self.T):
            dts = np.diag(D_t[i])
            dtinv = np.linalg.inv(dts)
            et = dtinv*self.rt[i].T
            Q_t[i] = (1-a-b)*Q_bar + a*(et*et.T) + b*Q_t[i-1]
            qts = np.linalg.inv(np.sqrt(np.diag(np.diag(Q_t[i]))))

            R_t[i] = np.matmul(qts, np.matmul(Q_t[i], qts))


            H_t[i] = np.matmul(dts, np.matmul(R_t[i], dts))   

            loglike = loglike + np.log( gamma((self.N+dof)/2.)) - np.log(gamma(dof/2)) \
                      -(self.N/2.)*np.log(np.pi*(dof - 2)) - np.log(np.linalg.det(H_t[i])) \
- ((dof+ self.N)*( ((np.matmul(self.rt[i], (np.matmul( np.linalg.inv(H_t[i]), self.rt[i].T))))/(dof - 2.)) + 1)/2.)


        return -loglike
    
    
    def predict(self, ndays = 1):
        if 'a' in dir(self):
            Q_bar = np.cov(self.rt.reshape(self.N, self.T))

            Q_t = np.zeros((self.T,self.N,self.N))
            R_t = np.zeros((self.T,self.N,self.N))
            H_t = np.zeros((self.T,self.N,self.N))

            Q_t[0] = np.matmul(self.rt[0].T/2, self.rt[0]/2)

            loglike = 0
            for i in range(1,self.T):
                dts = np.diag(self.D_t[i])
                dtinv = np.linalg.inv(dts)
                et = dtinv*self.rt[i].T
                Q_t[i] = (1-self.a-self.b)*Q_bar + self.a*(et*et.T) + self.b*Q_t[i-1]
                qts = np.linalg.inv(np.sqrt(np.diag(np.diag(Q_t[i]))))

                R_t[i] = np.matmul(qts, np.matmul(Q_t[i], qts))


                H_t[i] = np.matmul(dts, np.matmul(R_t[i], dts))  

            if self.dist == 'norm':
                return {'dist': self.dist, 'cov': H_t[-1]*np.sqrt(ndays)}
            elif self.dist == 't':
                return {'dist': self.dist, 'dof': self.dof, 'cov': H_t[-1]*np.sqrt(ndays)}
            
        else:
            print('Model not fit')
            
            
    def fit(self, returns):
        self.rt = np.matrix(returns)
        
        self.T = self.rt.shape[0]
        self.N = self.rt.shape[1]
        
        if self.N == 1 or self.T == 1:
            return 'Required: 2d-array with columns > 2' 
        self.mean = self.rt.mean(axis = 0)
        self.rt = self.rt - self.mean
        
        D_t = np.zeros((self.T, self.N))
        for i in range(self.N):
            params = self.garch_fit(self.rt[:,i])
            self.params = params
            D_t[:,i] = np.sqrt(self.garch_var(params, self.rt[:,i]))
        self.D_t = D_t
        if self.dist == 'norm':
            res = minimize(self.mgarch_loglike, (0.01, 0.94), args = D_t,
            bounds = ((1e-6, 1), (1e-6, 1)), 
            #options = {'maxiter':10000000, 'disp':True},
            )
            self.a = res.x[0]
            self.b = res.x[1]
            
            return {'mu': self.mean, 'alpha': self.a, 'beta': self.b} 
        elif self.dist == 't':
            res = minimize(self.mgarch_logliket, (0.01, 0.94, 3), args = D_t,
            bounds = ((1e-6, 1), (1e-6, 1), (3, None)), 
            #options = {'maxiter':10000000, 'disp':True},
            )
            self.a = res.x[0]
            self.b = res.x[1]
            self.dof = res.x[2]
            return {'mu': self.mean, 'alpha': self.a, 'beta': self.b, 'dof': self.dof} 

class CCC:
    
    def __init__(self):
        return
            
    def garch_fit(self, returns):
        res = minimize( self.garch_loglike, (0.01, 0.01, 0.94), args = returns,
              bounds = ((1e-6, 1), (1e-6, 1), (1e-6, 1)))
        return res.x

    def garch_loglike(self, params, returns):
        T = len(returns)
        var_t = self.garch_var(params, returns)
        LogL = np.sum(-np.log(2*np.pi*var_t)) - np.sum( (returns.A1**2)/(2*var_t))
        return -LogL

    def garch_var(self, params, returns):
        T = len(returns)
        omega = params[0]
        alpha = params[1]
        beta = params[2]
        var_t = np.zeros(T)     
        for i in range(T):
            if i==0:
                var_t[i] = returns[i]**2
            else: 
                var_t[i] = omega + alpha*(returns[i-1]**2) + beta*var_t[i-1]
        return var_t 
    
    def garch_predict(self,params,returns,ndays):
        T = len(returns)
        omega = params[0]
        alpha = params[1]
        beta = params[2]
        miu=np.mean(np.array(returns)**2)
        var_t=self.garch_var(params,returns)
        var_predict=[]
        for i in range(ndays):
            if i==0:
                pre=omega + alpha*(returns[-1]**2) + beta*var_t[-1]
                var_predict.append(pre[0,0])
            else:
                pre=omega + alpha*(miu) + beta*var_predict[i-1]
                var_predict.append(pre)
        return var_predict
    
    def predict(self, ndays = 1):
        var_predict=[]
        H_t = np.zeros((ndays,self.N,self.N))
        for i in range(self.N):
            var_predict.append(self.garch_predict(self.params[i],self.rt[:,i],ndays))
        var_predict=np.array(var_predict)
        for i in range(ndays):
            dts = np.diag(np.sqrt(var_predict[:,i]))
            R_t=self.R_t
            cov=np.matmul(dts, np.matmul(R_t, dts))
            H_t[i] = cov
        return H_t

            
    def fit(self, returns):
        self.rt = np.matrix(returns)
        self.R_t=np.corrcoef(returns.T)
        
        self.T = self.rt.shape[0]
        self.N = self.rt.shape[1]
        
        if self.N == 1 or self.T == 1:
            return 'Required: 2d-array with columns > 2' 
        self.mean = self.rt.mean(axis = 0)
        self.rt = self.rt - self.mean
        
        D_t = np.zeros((self.T, self.N))
        self.params=[]
        for i in range(self.N):
            params = self.garch_fit(self.rt[:,i])
            self.params.append(params)
            D_t[:,i] = np.sqrt(self.garch_var(params, self.rt[:,i]))
        self.D_t = D_t
    

class VECH:
    
    def __init__(self):
        return
            
    def vech_loglike(self, params,returns):
        # No of assets
        A = np.array([[params[0],params[1]],[params[1],params[2]]])
        B = np.array([[params[3],params[4]],[params[4],params[5]]])
        C = np.array([[params[6],params[7]],[params[7],params[8]]])
       
        H_t = np.zeros((self.T,self.N,self.N))
        et = np.array(self.rt[0].T)
        
        H_t[0]=np.matmul(et,et.T)
        loglike = 0
        for i in range(1,self.T):
            et = np.array(self.rt[i].T)
            H_t[i] = A*np.ones((2,2)) + B*np.matmul(et,et.T) + C*H_t[i-1]
            loglike = loglike + \
                      np.log(np.linalg.det(H_t[i])) + \
                      np.matmul(self.rt[i], (np.matmul( np.linalg.inv(H_t[i]), self.rt[i].T)))
        self.H_t=H_t
        self.miu=np.mean(et,axis=1)
        return loglike  
    
    def predict(self, ndays = 1):
        params=self.params
        A = np.array([[params[0],params[1]],[params[1],params[2]]])
        B = np.array([[params[3],params[4]],[params[4],params[5]]])
        C = np.array([[params[6],params[7]],[params[7],params[8]]])
        H_t = np.zeros((ndays,self.N,self.N))
        et = np.array(self.rt[-1].T)
        var_predict=[]
        for i in range(ndays):
            if i==0:
                H_t[i]=A*np.ones((2,2)) + B*np.matmul(et,et.T) + C*self.H_t[-1]
            else:
                et=self.miu
                H_t[i] = A*np.ones((2,2)) + B*np.matmul(et,et.T) + C*H_t[i-1]
        return H_t
            
            
    def fit(self, returns):
        self.rt = np.matrix(returns)
        
        self.T = self.rt.shape[0]
        self.N = self.rt.shape[1]
        
        if self.N == 1 or self.T == 1:
            return 'Required: 2d-array with columns > 2' 
        self.mean = self.rt.mean(axis = 0)
        self.rt = self.rt - self.mean

        res = minimize(self.vech_loglike, (0.0001,0.0001,0.0001,0.01,0.01,0.01,0.9,0.8, 0.9), args = self.rt,
        bounds = ((1e-6, 1), (1e-6,1), (1e-6, 1), (1e-6, 1), (1e-6, 1), (1e-6, 1), (1e-6, 1), (1e-6, 1), (1e-6, 1)),
        method="SLSQP"
            )
        self.params=res.x         
        return {"params":self.params} 
        
