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
    def __init__(self, Debounce=0):
        self.deb_class = ActiveDebouncer(threshold=Debounce)

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

    def tb_simulator_2_state_model(self, 
                                    t_body, t_pref_min, t_pref_max,
                                    burrow_temp, open_temp,
                                    t_crit_min=None, t_crit_max=None):
        if t_crit_min is None or t_crit_max is None:
            if t_body<=t_pref_min and open_temp>burrow_temp:
                # Leave Burrow to warm up
                t_env=open_temp #go to the warmest microhabitat
                bu = 'Out'
            elif t_body>=t_pref_max and open_temp>burrow_temp:
                # go into cool mh to cool down
                t_env=burrow_temp
                bu = 'In'
            elif t_body<=t_pref_min and open_temp<burrow_temp:
                t_env=burrow_temp
                bu = 'In'
            else:
                t_env=open_temp
                bu = 'Out'
        else:
            if t_body>=t_crit_max:
                # go into burrow youre about to burn to death
                t_env=burrow_temp
                bu = 'In'
            elif t_body<=t_crit_min:
                # go into burrow youre about to freeze to death
                t_env=burrow_temp
                bu = 'In'
            elif t_body<=t_pref_min and open_temp>burrow_temp:
                # Leave Burrow to warm up
                t_env=open_temp #go to the warmest microhabitat
                bu = 'Out'
            elif t_body>=t_pref_max and open_temp>burrow_temp:
                # go into cool mh to cool down
                t_env=burrow_temp
                bu = 'In'
            elif t_body<=t_pref_min and open_temp<burrow_temp:
                # Its too cold out, go into burrow
                t_env=burrow_temp
                bu = 'In'
            else:
                t_env=open_temp
                bu = 'Out'
        bu = self.deb_class.debounce_alg(input_state = bu)
        if bu=='In':
            t_env=burrow_temp
        else:
            t_env=open_temp
        return bu, t_env

    def tb_simulator_2_state_model_wrapper(self, 
                                           k, t_initial,
                                           t_pref_min, t_pref_max,
                                            burrow_temp_vector, open_temp_vector,
                                           t_crit_min=None, t_crit_max=None,
                                           return_tbody_sim=False):
        simulated_t_body = []
        burrow_usage = []
        t_body=t_initial
        time=0
        for index, (burrow_temp, open_temp) in enumerate(zip(burrow_temp_vector, open_temp_vector)):
            bu, t_env = self.tb_simulator_2_state_model(t_body=t_body, t_pref_min=t_pref_min, t_pref_max=t_pref_max, burrow_temp=burrow_temp, open_temp=open_temp)
            simulated_t_body.append(t_body)
            burrow_usage.append(bu)
            t_body = self.cooling_eq_k(k=k, t_body=t_body, t_env=t_env, time=time)
            time+=1
        if return_tbody_sim:
            return burrow_usage, simulated_t_body
        else:
            return burrow_usage

    # def tb_simulator_n_state_model(self, 
    #                                k, t_initial,
    #                                t_pref_min, t_pref_max, burrow_temp_vector,
    #                                t_crit_min=None, t_crit_max=None,
    #                                return_tbody_sim=False, return_microhabitat=False, **t_env_vectors,):
    #     simulated_t_body = []
    #     burrow_usage = []
    #     microhabitat = []
    #     df = pd.DataFrame.from_dict(t_env_layers)
    #     df['Burrow_Temp'] = burrow_temp_vector
    #     t_body=t_initial
    #     time=0
    #     for index, row in df.iterrows():
    #         min_outside_env_index = row.loc[:, row.columns != 'Burrow_Temp'].idxmin()
    #         min_outside_env_val = row[min_outside_env_index]
    #         max_outside_env_index = row.loc[:, row.columns != 'Burrow_Temp'].idxmax()
    #         max_outside_env_val = row[max_outside_env_index]
    #         if t_crit_min is None or t_crit_max is None:
    #             pass

    #         else:
    #             if t_body>=t_crit_max:
    #                 # go into burrow youre about to burn to death
    #                 t_env=row['Burrow_Temp']
    #                 bu = 'In'
    #             elif t_body<=t_crit_min:
    #                 # go into burrow youre about to freeze to death
    #                 t_env=row['Burrow_Temp']
    #                 bu = 'In'
    #             elif t_body<=t_pref_min and open_temp>burrow_temp:
    #                 # Leave Burrow to warm up
    #                 t_env=max_outside_env_val #go to the warmest microhabitat
    #                 bu = 'Out'
    #                 mh = row.columns[max_outside_env_index]
    #             elif t_body>=t_pref_max and open_temp>burrow_temp:
    #                 # go into cool mh to cool down
    #                 t_env=row['Burrow_Temp']
    #                 bu = 'In'
    #             elif t_body<=t_pref_min and open_temp<burrow_temp:
    #                 t_env=row['Burrow_Temp']
    #                 bu = 'In'
    #             else:
    #                 t_env=open_temp
    #                 bu = 'Out'
    #                 # forage
    #         simulated_t_body.append(t_body)
    #         burrow_usage.append(bu)
    #         t_body = self.cooling_eq_k(k=k, t_body=t_body, t_env=t_env, time=time)
    #         #t_body = round(t_body + rate_of_heat_transfer_k(k=k, t_body=t_body, t_env=t_env),5)
    #         time+=1
        


if __name__ ==  "__main__":
    pass