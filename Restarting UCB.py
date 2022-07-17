import requests
import math
import json
import csv
from pprint import pprint
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
import random
import math
import time
import seaborn as sns
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter1d
import re
from scipy import stats
from scipy.optimize import minimize
from scipy import optimize



class UCB1():
    
    def __init__(self, c=1):
        
        self.c = c
        self.actions = 5
        
        self.counts = [0 for col in range(self.actions)]
        
        self.values = [0.0 for col in range(self.actions)]
        
        self.action_total_reward = [0.0 for _ in range(self.actions)]
        self.action_avg_reward = [[] for action in range(self.actions)]
        
        return
    def max_(self,values):
        max_index = 0
        maxv = values[max_index]
        for i in range(len(values)):
            if values[i] > maxv:
                maxv = values[i]
                max_index = i
        return max_index

    def select_action(self):
        actions = len(self.counts)
        for action in range(actions):
            if self.counts[action] == 0:
                return action
    
        ucb_values = [0.0 for action in range(actions)]
        total_counts = sum(self.counts)
        for action in range(actions):
            bonus =  EXP_RATE * (math.sqrt((2 * math.log(total_counts)) / float(self.counts[action])))
            ucb_values[action] = self.values[action] + bonus
        return self.max_(ucb_values)
    def update(self, chosen_act, reward):
        self.counts[chosen_act] = self.counts[chosen_act] + 1
        n = self.counts[chosen_act]
        
#     # Update average/mean value/reward for chosen action
        value = self.values[chosen_act]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        #new_value2 = value + (1/n * (reward - value))
        self.values[chosen_act] = new_value
        
        self.action_total_reward[chosen_act] += reward
        for a in range(self.actions):
            if self.counts[a]:
                self.action_avg_reward[a].append(self.action_total_reward[a]/self.counts[a])
            else:
                self.action_avg_reward[a].append(0)
        return
    

def simulation_constant_max_c_0_1_opt(mus, sigmas, n_trails,Max_val,c): 
    
    ob=UCB1(c=c)
    
    
    
   
   
    for i in range(n_trails):
    
        
    
        #reward = random.random()
        action= ob.select_action()
        selected_actions.append(action)
        #print(action)
        #print(ob.actions[action])
        
        mu, sigma = mus[action], sigmas[action]
        #time.sleep(1)
        delay = np.random.normal(mu, sigma)
                
        
        cost = delay/Max_val
        x1 = delay
        reward = 1-cost
        #print(f" chosen action: {action},reward: {reward}")
        
        rewards.append(reward)
        res.append(x1)
        
        
        cum_response_time.append(sum(res)/len(res))
        
        
        ob.update(action, reward)
        #print(f"Q_table_NU: {ob.values}")
        
    return rewards, cum_response_time, ob.counts, ob.values, delay, res,selected_actions

if __name__ == '__main__':
    
    cum_response_time = []
    rewards = []
    res = [] # contains all delays
    #Max_val = 1.0 
    selected_actions = []
    
    
    mus = [0.39, 0.35, 0.44, 0.36, 0.47]
    sigmas = [0.18,0.13,0.13,0.2,0.26]
    n_trails = 100
    n_sim = 1
    #Norm_cons = [0.1, 0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8, ]
    Norm_cons = 0.2
    ex_cons = list(np.arange(0.1, 1.1, 0.1))
    results = {}
    all_coeff = []
    all_res = []
    
    
    
    
    for ex in ex_cons:
        EXP_RATE = ex
        #print(f'EXP_RATE = {EXP_RATE}---------------\n')
        rewards_NU = []
        cum_response_time_NU = []
        counts_NU = []
        values_NU = []
        delay_NU = []
        res_NU = [] # raw reward
        best_action = []
        
        results[ex]= {}
        
        for sim in tqdm(range (n_sim)):
            rewards_N, cum_response_time_N, counts_N, values_N, delay_N, res_N, selected_action =\
                simulation_constant_max_c_0_1_opt(mus, sigmas, n_trails, Norm_cons,ex)
            rewards_NU.append(rewards_N)
            cum_response_time_NU.append(cum_response_time_N)
            counts_NU.append(counts_N)
            values_NU.append(values_N)
            delay_NU.append(delay_N)
            res_NU.append(res_N)
            best_action.append(selected_action)
        results[ex] = np.array(cum_response_time_NU)
        
    
    avg_resp_time = {}
    for iteration in range(10):
        
        for ex in ex_cons:
            
            avg_resp_time[ex]= results[ex].mean()
        #fitting the model
        poly_fit= np.poly1d(np.polyfit(list(avg_resp_time.keys()), list(avg_resp_time.values()), 2))
        all_coeff.append(poly_fit)
        def resp_time(c):
            return poly_fit[0]* c**2 + poly_fit[1]* c + poly_fit[2]
    
        minimize_f = optimize.minimize_scalar(fun= resp_time, bounds = (0,1),method='Bounded')
        min_c = minimize_f.x
        ex_cons.append(min_c)
        
        results[min_c]= {}
        
        for sim in tqdm(range (n_sim)):
            EXP_RATE= min_c
            #print(f'new Exp_rate = {EXP_RATE}...........\n')
            rewards_N, cum_response_time_N, counts_N, values_N, delay_N, res_N,selected_action = simulation_constant_max_c_0_1_opt(mus, sigmas, n_trails, Norm_cons,min_c)
            rewards_NU.append(rewards_N)
            cum_response_time_NU.append(cum_response_time_N)
            counts_NU.append(counts_N)
            values_NU.append(values_N)
            delay_NU.append(delay_N)
            res_NU.append(res_N)
            best_action.append(selected_action)
        results[min_c] = np.array(cum_response_time_NU)
        
        
        
    plt.figure(1,dpi=120)
    plt.plot(list(avg_resp_time.values()), 'o--')
    plt.xlabel('Improvment'   , fontsize=10)
    plt.ylabel('Average Response', fontsize=10)
    plt.grid(True)
    plt.show()
    
    plt.figure(1,dpi=120)
    best_action = np.array(best_action)
    r = (best_action == 1).cumsum(axis=1) / np.arange(1, len(selected_action)+1)
    plt.plot(r.mean(axis=0), label='Restarting UCB', ls='--')
    #plt.fill_between(np.arange(1,1401), r.mean(axis=0) - r.std(axis=0),r.mean(axis=0) + r.std(axis=0), alpha=0.3 );
    plt.legend(bbox_to_anchor=(1., 1.))
    plt.xlabel('Time steps', fontsize=10)
    plt.ylabel('Probability of Selecting Best Action', fontsize=10)
    plt.grid(True)
    plt.show()
    
    plt.figure(1,dpi=120)
    # learning curve for all C values just for clarification  
    for ex in ex_cons:
        plt.plot(results[ex].mean(axis=0), label=f'C={ex}')
        
    plt.legend(bbox_to_anchor=(1., 1.))
    plt.title('Average Response when normalization value=0.2' , fontsize=10)
    plt.xlabel('Time steps'   , fontsize=10)
    plt.ylabel('Average Response', fontsize=10)
    plt.grid(True)
    plt.show()
    
    plt.figure(1,dpi=120)
    # The final learning curve
    plt.plot(results[ex].mean(axis=0))
    plt.title('Average Response when normalization value=0.2' , fontsize=10)
    plt.xlabel('Time steps'   , fontsize=10)
    plt.ylabel('Average Response', fontsize=10)
    plt.grid(True)
    plt.show()

