# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 21:41:39 2022

@author: XYZ
"""

#%%
print('Running...')

#
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import functools
import os, pickle

#
sec=1
um=1
uM=1

#%%
class VF_model(object):
    def __init__(self):
        # simulation config
        self.nSteps                         = 3000
        
        # a virtual bacterium
        self.tau_ccw0                       = 1.75 *(sec)
        self.tau_cw0                        = 3.50 *(sec)
        self.Vp                             = 16.0 *(um/sec)                    # pushing speed
        self.Vw                             = 8.0 *(um/sec)                     # wrapping speed
        self.start_pos                      = [4000.0,0.0]
        
        # chemotaxis source
        self.conc_max                       = 10000 *(uM)
        self.conc_center                    = [0.0,0.0] *(um)
        self.conc_sigma                     = 200
        
        # sensing
        self.K                              = 3.0 *(1/uM)
        self.sense_conc_low                 = 0.1 *(uM)
        self.sense_conc_high                = 100 *(uM)
        
        # 
        self.sync_prob0                     = 0.7

    def dist_calc(self,pos1,pos2):
        return math.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)

    def conc_calc(self,pos):
        dist=self.dist_calc(pos,self.conc_center)
        conc=self.conc_max*math.exp(-(dist/(math.sqrt(2)*self.conc_sigma))**2)
        return conc
  
    def path_generator(self,nCell,is_che,is_high_sync):
        # SEED = nCell
        # np.random.seed(SEED)
        
        # initial conditions ##################################################
        nSteps=self.nSteps
        tau_ccw0=self.tau_ccw0
        tau_cw0=self.tau_cw0
        Vp=self.Vp
        Vw=self.Vw
        start_pos=self.start_pos
        
        # given an initail particle ###########################################
        S=np.array([0])                                                         # record swimming state [0: push, 1: wrap]
        R=np.array([0])                                                         # 0:CCW, 1: CW
        T=np.array([np.random.exponential(tau_ccw0)])                           # record the period time of motile state
        curr_pos=np.array(start_pos)
        curr_dir=np.random.rand()*2*math.pi
        
        if is_che==False:
            if is_high_sync:
                sync_prob=self.sync_prob0
            else:
                sync_prob=0.01
        else:
            curr_conc=self.conc_calc(curr_pos)
            if curr_conc<self.sense_conc_low or curr_conc>self.sense_conc_high:
                if is_high_sync==True:
                    sync_prob=self.sync_prob0
                else:
                    sync_prob=0.01
            else:
                sync_prob=0.01
        
        # initial variables ###################################################
        path=curr_pos[np.newaxis]
        
        # simulation ##########################################################
        for nStep in range (1,nSteps):
            if S[nStep-1]==0: # if previous swimming state is push, then current state is wrap (CW)
                S=np.append(S,[1])
                R=np.append(R,[1])
            else: # if previous swimming state is wrap, then current state is wrap or push
                if np.random.rand()<sync_prob:
                    S=np.append(S,[1])
                    if R[nStep-1]==0:
                        R=np.append(R,[1])
                    else:
                        R=np.append(R,[0])
                else:
                    S=np.append(S,[0])
                    R=np.append(R,[0])
            
            # determine current speed
            if S[nStep]==0:
                speed=Vp
            else:
                speed=Vw
            
            # determine current direction
            if S[nStep-1]+S[nStep]==1:
                th=math.pi-np.random.exponential(math.pi/6)
                curr_dir=curr_dir+np.random.choice(np.array([-1,1]))*th
            
            # determine current tau
            if R[nStep]==0:
                if is_che==False:
                    T=np.append(T,np.random.exponential(tau_ccw0))
                else:
                    curr_conc=self.conc_calc(curr_pos)
                    tau_ccw=tau_ccw0*(4/(1+3*math.exp(-self.K*curr_conc)))
                    future_pos=curr_pos+speed*tau_ccw*np.array([np.cos(curr_dir),np.sin(curr_dir)])
                    future_conc=self.conc_calc(future_pos)
                    diff_conc=future_conc-curr_conc
                    if diff_conc<0 and curr_conc>self.sense_conc_low:
                        tau_ccw=0.4 *(sec)                                      # response to chemotaxis change
                    T=np.append(T,np.random.exponential(tau_ccw))
            else:
                T=np.append(T,np.random.exponential(tau_cw0))

            # update current position
            curr_pos=curr_pos+speed*T[nStep]*np.array([np.cos(curr_dir),np.sin(curr_dir)])
            path=np.append(path,curr_pos[np.newaxis],axis=0)   
            
            #
            if is_che==True:
                if curr_conc<self.sense_conc_low or curr_conc>self.sense_conc_high:
                    if is_high_sync==True:
                        sync_prob=self.sync_prob0
                    else:
                        sync_prob=0.01                                          # if compare effect, modify this value to 0.01     
                    if curr_conc>self.sense_conc_high:
                        tau_ccw=tau_ccw0
                else:
                    sync_prob=0.01
        
        # pushing period
        T_p=[]
        for nStep in range(nSteps):
            if S[nStep]==0:
                T_p.append(T[nStep])

        # wrapping period
        T_w=[]
        T_w_=0
        for nStep in range(nSteps):
            if S[nStep]==1:
                T_w_=T_w_+T[nStep]
            else:
                T_w.append(T_w_)
                T_w_=0   
        return S,R,T,path,T_p,T_w
    
    def paths_plot(self,bac_list,is_all):
        nCells=len(bac_list)
        nframe_plot=100
        if is_all==True:
            plt.figure()
            # draw trajectories
            for nCell in range(nCells):
                path=bac_list[nCell][3]
                plt.plot(path[:,0],path[:,1],'gray',lw=0.3)
    
            # draw statpoint and goal
            for nCell in range(nCells):
                path=bac_list[nCell][3]
                plt.plot(path[0,0],path[0,1],'r>',path[-1,0],path[-1,1],'ks',markersize=2)
                plt.grid(lw=0.3)
                plt.xlim((-2000,8000))
                plt.ylim((-5000,5000))
                plt.gca().set_aspect('equal')
            plt.show()
        else:
            for nT in range(int(np.shape(bac_list[0][3])[0]/100)):
                plt.figure()
                for nCell in range(nCells):
                    path=bac_list[nCell][3]
                    plt.plot(path[0:nframe_plot*(nT+1),0],path[0:nframe_plot*(nT+1),1],'gray',lw=0.3)
                
                for nCell in range(nCells):
                    path=bac_list[nCell][3]
                    plt.plot(path[0,0],path[0,1],'r>',path[nframe_plot*(nT+1)-1,0],path[nframe_plot*(nT+1)-1,1],'ks',markersize=2)
                    plt.grid(lw=0.3)
                    plt.xlim((-2000,8000))
                    plt.ylim((-5000,5000))
                    plt.gca().set_aspect('equal')
                plt.show()
    
    def drift_dir_calc(self,bac_list):
        nCells=len(bac_list)
        th=[]
        drift_dist=[]
        for nCell in range(nCells):
            path=bac_list[nCell][3]
            dx=path[-1,0]-self.start_pos[0]
            dy=path[-1,1]-self.start_pos[1]
            th.append(np.arctan2(dy,dx)*180/math.pi)
            drift_dist.append(math.sqrt(dx**2+dy**2))
        # print('\t The mean orientation of drift: %s [deg].' %(np.mean(th)))
        # print('\t The mean drift displacement: %s [um].' %(np.mean(drift_dist)))
        return np.mean(drift_dist)
        
    def arrival_calc(self,bac_list):
        nCells=len(bac_list)
        nArrivals=0
        for nCell in range(nCells):
            path=bac_list[nCell][3]
            dx=path[-1,0]-self.conc_center[0]
            dy=path[-1,1]-self.conc_center[1]
            dr=math.sqrt(dx**2+dy**2)
            
            if dr<math.sqrt(-2*(self.conc_sigma**2)*math.log(self.sense_conc_low/self.conc_max)):
                nArrivals=nArrivals+1
        # print('\t Arrival cells / Total cells: %d / %d' %(nArrivals,nCells))
        # print('\t The arrival efficiency: %s' %(nArrivals/nCells))
        return nArrivals/nCells
        
    def T_calc(self,bac_list):
        nCells=len(bac_list)
        T_p_sum=0
        T_p_num=0
        T_w_sum=0
        T_w_num=0
        T_ccw_sum=0
        T_ccw_num=0
        T_cw_sum=0
        T_cw_num=0
        num_push=0
        num_wrap=0
        for nCell in range(nCells):
            S_=bac_list[nCell][0]
            num_wrap=num_wrap+np.sum(S_)
            num_push=num_push+len(S_)-np.sum(S_)
            
            T_p_=bac_list[nCell][4]
            T_w_=bac_list[nCell][5]
            T_p_num=T_p_num+len(T_p_)
            T_w_num=T_w_num+len(T_w_)
            T_p_sum=T_p_sum+np.sum(T_p_)
            T_w_sum=T_w_sum+np.sum(T_w_)
            
            T_=bac_list[nCell][2]
            R_=bac_list[nCell][1]
            T_ccw_sum=T_ccw_sum+np.sum(T_[R_==0])
            T_ccw_num=T_ccw_num+len(T_[R_==0])
            T_cw_sum=T_cw_sum+np.sum(T_[R_==1])
            T_cw_num=T_cw_num+len(T_[R_==1])
        # print('\t The mean time of pusher: %s [sec].' %(T_p_sum/T_p_num))
        # print('\t The mean time of wrapper: %s [sec].' %(T_w_sum/T_w_num))
        # print('\t The occupancy of pusher: %s' %(num_push/(num_push+num_wrap)))
        # print('\t The mean time of ccw: %s [sec].' %(T_ccw_sum/T_ccw_num))
        # print('\t The mean time of cw: %s [sec].' %(T_cw_sum/T_cw_num))

#%%
if __name__ == '__main__':
    VF=VF_model()
    
    # set simulation condition
    nCells = 1000
    nLoops = 20
    
    list_sync_prob0 = [0.01,0.3,0.5,0.7,0.75,0.8,0.85,0.9,0.95,0.99,0.999]
    list_nSteps = [1000,3000,5000,7000,9000,12000,15000,20000,25000,30000,50000]
    
    # initialize
    map_sync_prob0 = np.tile(list_sync_prob0,[len(list_nSteps),1])
    map_nSteps = np.tile(list_nSteps,[len(list_sync_prob0),1]).T
    loop_msd = np.zeros((nLoops,len(list_nSteps),len(list_sync_prob0)))
    loop_arrival = np.zeros((nLoops,len(list_nSteps),len(list_sync_prob0)))
    nRuns=nLoops*len(list_nSteps)*len(list_sync_prob0)
    nRun=0
    
    # simulation
    print('> Start simulating a bacterial swimming process with high synchronization under gradient environment...')
    pool = multiprocessing.Pool()
    for nLoop in range(nLoops):
        tic=time.time()
        for row in range(len(list_nSteps)):
            for col in range(len(list_sync_prob0)):
                print("Running...(%.2f%%)" %(100*nRun/nRuns))
                
                # set parameters
                VF.nSteps=map_nSteps[row,col]
                VF.sync_prob0=map_sync_prob0[row,col]
                
                # simulate a bacterial swimming process
                cells4=pool.map(functools.partial(VF.path_generator,is_che=True,is_high_sync=True),range(nCells))
                
                # analyze trjaectories
                VF.paths_plot(cells4,is_all=True)
                loop_msd[nLoop,row,col]=VF.drift_dir_calc(cells4)
                loop_arrival[nLoop,row,col]=VF.arrival_calc(cells4)
                VF.T_calc(cells4)
                
                # counting
                nRun=nRun+1
        print("--- (Simulation time: %s seconds) ---" % (time.time()-tic))
        
        # save simulation results
        print("Start saving simulation results...")
        outputfile=os.getcwd()+'\\VF_che_simulation_V12_nLoops_'+str(nLoops)+'_nCells_'+str(nCells)+'_DeltaR_4000'
        Results=[map_sync_prob0,map_nSteps,loop_msd,loop_arrival]
        with open(outputfile,'wb') as f: pickle.dump(Results,f)
    pool.close()
    pool.join()

#%%
print('Done.')
print('-------------------------------------------------------------------')