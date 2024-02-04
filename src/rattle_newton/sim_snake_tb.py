#!/usr/bin/python
import pandas as pd
import numpy as np
import math

class ActiveDebouncer(object):
    def __init__(self, threshold=3):
        self.threshold = threshold  # Threshold for stable state
        self.current_state = None # Initial state
        self.transition_count = 0  # Counter for state transitions

    def debounce_alg(self, input_state):
        if self.current_state is None:
            # start the alg
            self.current_state = input_state
        elif input_state != self.current_state:
            if self.transition_count < self.threshold:
                # Ignore rapid state changes
                self.transition_count += 1
            else:
                # update states
                self.current_state = input_state
                self.transition_count = 0
                return self.current_state
        else:
            # No change in input state
            pass
        return self.current_state


class ThermalSimulator(object):
    def __init__(self, flip_logic, t_pref_min, t_pref_max, t_pref_opt, seed=None, Debounce=0):
        self.deb_class = ActiveDebouncer(threshold=Debounce)
        self.flip_logic = flip_logic
        self.t_pref_min = t_pref_min
        self.t_pref_max = t_pref_max
        self.t_pref_opt = t_pref_opt
        self.rng = np.random.default_rng(seed)
        self.state_switch = None
        self.current_state = None


    def rate_of_heat_transfer_k(self, k, t_body, t_env):
        return k*(t_body-t_env)

    def cooling_eq_k(self, k, t_body, t_env, time):
        return t_env+(t_body-t_env)*math.exp(-k*time)

    def k_approximation(self,t_body, t_body_prev, tenv_prev,  time):
        k = math.log(abs((t_body - tenv_prev) / (t_body_prev  - tenv_prev))) / time
        return k

    def snake_surface_area(self,radius, length):
        return 2*math.pi*radius*length+2*math.pi*radius**2

    def rate_of_heat_transfer_h(self,h, sa, t_body, t_env):
        '''Newtons law of cooling equation with h coeffiecent and surface area rather than using k. This function calculates the rate of change of tb.'''
        return h*sa*(t_body-t_env)

    def coolong_eq_h(self,h, sa, t_body, t_env, time):
        '''Newtons law of cooling equation with h coeffiecent and surface area rather than using k. This function calculates tb.'''
        return t_env+(t_body-t_env)*math.exp(h*sa*time)

    def h_approximation(self,t_body, t_body_prev, tenv_prev, sa, time):
        h = math.log(abs((t_body-tenv_prev)/(t_body_prev - tenv_prev)))/(sa*time)
        return h

    def cooling_or_heating(self,t_body, t_body_prev):
        if round(t_body-t_body_prev,3)>0:
            val='Increasing'
        elif round(t_body-t_body_prev,3)<0:
            val='Decreasing'
        else:
            val='Constant'
        return val

    def random_flips(self):
        if self.rng.random() <= 0.5:
            bu = 'In'
        else:
            bu = 'Out'
        return bu

    def preferred_topt(self, t_body, burrow_temp, open_temp):
        if t_body >= self.t_pref_opt:
            prob_flip =  (t_body - self.t_pref_opt) / (self.t_pref_max - self.t_pref_opt)
            if burrow_temp < open_temp:
                flip_direction = 'In'
            else:
                flip_direction = 'Out'
        elif t_body < self.t_pref_opt:
            prob_flip =  (self.t_pref_opt - t_body) / (self.t_pref_opt - self.t_pref_min)
            if burrow_temp > open_temp:
                flip_direction = 'In'
            else:
                flip_direction = 'Out'
        if self.rng.random() <= prob_flip:
            bu = flip_direction
            self.state_switch = 'Switch'
        elif self.state_switch is None:
            bu = self.random_flips()
            self.current_state = bu
        else:
            self.state_switch == 'Stay'
            bu = self.current_state
        return bu

    def boundary_tpref(self, burrow_temp, open_temp):
        if t_body<=self.t_pref_min and open_temp>burrow_temp:
            # Leave Burrow to warm up
            t_env=open_temp #go to the warmest microhabitat
            bu = 'Out'
        elif t_body>=self.t_pref_max and open_temp>burrow_temp:
            # go into cool mh to cool down
            t_env=burrow_temp
            bu = 'In'
        elif t_body<=self.t_pref_min and open_temp<burrow_temp:
            t_env=burrow_temp
            bu = 'In'
        else:
            t_env=open_temp
            bu = 'Out'
        return bu


    def do_i_flip(self, t_body, burrow_temp, open_temp):
        if self.flip_logic == 'random':
            bu = self.random_flips()
        elif self.flip_logic == 'preferred':
            bu = self.preferred_topt(t_body = t_body, burrow_temp = burrow_temp, open_temp = open_temp)
        elif self.flip_logic == 'boundary':
            bu = self.boundary_tpref(t_body = t_body, burrow_temp = burrow_temp, open_temp = open_temp)
        else:
            raise ValueError(f'No {self.flip_logic} defined')
        return bu


    def tb_simulator_2_state_model_wrapper(self, 
                                           k, t_initial,
                                            burrow_temp_vector, open_temp_vector,
                                           t_crit_min=None, t_crit_max=None,
                                           return_tbody_sim=False):
        simulated_t_body = []
        burrow_usage = []
        t_body=t_initial
        time=0
        for index, (burrow_temp, open_temp) in enumerate(zip(burrow_temp_vector, open_temp_vector)):
            bu = self.do_i_flip(t_body = t_body, burrow_temp = burrow_temp, open_temp = open_temp)
            if bu=='In':
                t_env=burrow_temp
            else:
                t_env=open_temp
            simulated_t_body.append(t_body)
            burrow_usage.append(bu)
            #self.current_bu = bu
            t_body = self.cooling_eq_k(k=k, t_body=t_body, t_env=t_env, time=time)
            time+=1
        if return_tbody_sim:
            return burrow_usage, simulated_t_body
        else:
            return burrow_usage


if __name__ ==  "__main__":
    pass