# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 09:34:28 2022

@author: Abdhul Khadhir
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import math
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize, Bounds, brute



# %% Constants initialisation
n_lanes = {'EB': 2, 'WB': 2, 'NB': 2, 'SB' : 2}
n_actual = {}
n_virtual = {}
pce_used = {}
vehs = {}
vehs_pce_default = {1:1900, 2:800, 3:1500, 4:900}
cars = {}
cars_default = {1:900, 2:350, 3:700, 4:400}
two_wheelers = {}
two_wheelers_default = {1:950, 2:450, 3:800, 4:500}
three_wheelers = {}
three_wheelers_default = {1:100, 2:34, 3:65, 4:36}
heavy_vehicles = {}
heavy_vehicles_default = {1:20, 2:10, 3:15, 4:12}
greens = {}
greens_default = {1:22, 2:9, 3:18, 4:11}
delays = {}
s = {}

# %% Functions

# Approach Delay
def approach_delay(C,g,s,v,n):
    lam = g/C
    X = v/(s*lam)
    d1 = 0.5*C*(((1-lam)**2)/(1-v/s))
    d2 = (X**(math.sqrt(2*(n+1)))) / (2*n*lam*(1-X))
    if d2 < 0:
        d2 = np.inf
    d = d1+d2+4.84*lam-13.15
    return d

# Webster Timings
def webster_timings(demand):
    Y = demand.v_s.sum()
    L = len(demand)*l
    C = math.ceil((1.5*L + 5) / (1-Y))
    eff_g = C - L
    demand['greens_actual'] =round( (eff_g/Y) * demand.v_s )
    demand['greens_limitted'] = [max(min(x, 201), 8) for x in demand['greens_actual']]
    return list(demand.greens_limitted)

# Objective function
def obj_function(greens):
    #print(str(greens[0]) + " : "+str(greens[1]) + " : "+str(greens[2]) + " : "+str(greens[3]))
    #print(str(greens[0]) + " : "+str(greens[1]))
    avg_delay = []
    tot_delay = []
    C = sum(greens) + len(demand)*l
    z = 0
    for i in range(0, len(demand)):
        d_i = approach_delay(C,greens[i], demand.s[i], demand.v[i], demand.n[i])
        if d_i <= 0:
            avg_delay.append(np.inf)
            tot_delay.append(np.inf)
        else:
            avg_delay.append(d_i)
            tot_delay.append(d_i*demand.v[i])
    demand['avg_delay'] = avg_delay
    demand['tot_delay'] = tot_delay
    z= demand.tot_delay.sum()/demand.v.sum()
    return z

#%%

# st.set_page_config(layout='wide')
#Title
st.title('Optimal Signal Control and Analysis for Heterogeneous and Lane-Free Traffic (OSCAH)')

with st.expander('Input Parameters'):
    with st.container():
        st.header('Field Calibration Constants')
        # st.subheader('Intersection specific constants')

        col1, col2 = st.columns(2)
        # %% Sat flow
        with col1:
            sat_flow_rate = float(st.text_input("Enter saturation flow rate in PCE/hr/lane (if known)",
                                          2900, key = 'sat_flow_input'))
        
        # %% Actual lanes
        with col2:
            n_appr = int(st.text_input('Enter the number of intersection approaches', 4, key = 'n_appr_input'))
        
        with st.container():
            st.write("________________________________________________________________________________________")
            st.write("Enter the number of lanes in each Approach")
            lanes_columns = st.columns(n_appr)
            for appr in range(1, n_appr+1):
                with lanes_columns[appr-1]:
                    n_actual[appr] = int(st.text_input(f'Approach {appr}: ', 3,
                                                   key = 'n_actual_'+str(appr)))
                    s[appr] = sat_flow_rate*n_actual[appr]
        
    
    # %% Virtual lanes
    with st.container():
        st.write("________________________________________________________________________________________")
        st.write("Do you know the number of parallel movmeents / virtual lanes present in each approach? \
                 \n If unknown, please leave the checkbox unselected. A default value of number of actual lanes plus 2 will be\
                     taken as the number of virtual lanes")
        n_virt_enter = st.checkbox('Number of parallel movements / virtual lanes known?')
        
        if n_virt_enter:
            st.write('Enter the number of virtual lanes / parallel movements in each approach')
            virtual_lanes_columns = st.columns(n_appr)
            for appr in range(1, n_appr+1):
                with virtual_lanes_columns[appr-1]:
                    n_virtual[appr] = int(st.text_input(f'Approach {appr}: ', 5,
                                                   key = 'n_virtual_'+str(appr)))
        else:
            for appr in range(1, n_appr+1):
                n_virtual[appr] = n_actual[appr] + 2
            
    # %% PCE 
    pce_df = {'Vehicle Type': ['Car', 'Two Wheeler', 'Three Wheeler', 'Heavy Vehicles'],
           'PCE Value': [1,0.78,1.92,3.42]}
    pce_df = pd.DataFrame(pce_df)
    pce = {'Two-wheeler': 0.78, 'Three-wheeler': 1.92, 'Heavy-vehicles': 3.42}
    st.write("________________________________________________________________________________________")
    st.write("Do you know the Passenger Car Equivalent Values of all the vehicle types?")
    st.write("if unknown, the folowing default values will be used: ")
    st.subheader('Default PCE values')
    pc_cols = st.columns(4)

    pc_cols[0].metric("Car", 1.00)
    pc_cols[1].metric("Two-Wheeler", 0.78)
    pc_cols[2].metric("Three-Wheeler", 1.92)
    pc_cols[3].metric("Heavy Vehicle", 3.42)
        # st.table(pce_df)
    
    pce_enter = st.checkbox('PCE values known?')
    if pce_enter:
        st.write("Enter desired PCE values")
        p_col = st.columns(3)
        cou = 0
        for veh in pce:
            with p_col[cou]:
                pce_used[veh] = float(st.text_input(f'{veh}: ', pce[veh],
                                               key = 'pce_user_'+str(veh)))
            cou = cou+1
    else:
        pce_used = pce

    # %% Vehicle demand
    st.write("________________________________________________________________________________________")
    st.header('Traffic Demand and Signal Timiings')
    st.subheader("Traffic Demand Inputs")
    st.write("Do you know the approachwise demand in PCE/hr? If unknown, please leave the checkbox unselected and you will be prompted to enter class specific vehicle counts")
    demand_pce_enter = st.checkbox('Demand values in PCE known?', True)
    
    if demand_pce_enter:
        st.write("Enter the traffic demand in PCE/hr for each approach")
        pce_appr_columns = st.columns(n_appr)
        for appr in range(1, n_appr+1):
            with pce_appr_columns[appr-1]:
                vehs[appr] = float(st.text_input(f'Approach {appr}: ',
                                               vehs_pce_default[appr], key = 'vehs_pce_'+str(appr)))
    else:
        for appr in range(1, n_appr+1):
            st.write(f"Enter number of vehicles in each class for approach {appr}")
            class_appr_columns = st.columns(4)
            with class_appr_columns[0]:
                cars[appr] = int(st.text_input('Cars: ', 
                                               cars_default[appr], key = 'cars_pce_'+str(appr)))
            with class_appr_columns[1]:
                two_wheelers[appr] = int(st.text_input('Two wheelers: ', 
                                               two_wheelers_default[appr], key = 'two_wheelers_pce_'+str(appr)))
            with class_appr_columns[2]:
                three_wheelers[appr] = int(st.text_input('Tthree wheelers: ', 
                                               three_wheelers_default[appr], key = 'three_wheelers_pce_'+str(appr)))
            with class_appr_columns[3]:
                heavy_vehicles[appr] = int(st.text_input('Heavy vehicles: ', 
                                               heavy_vehicles_default[appr], key = 'heavy_vehicles_pce_'+str(appr)))
                
            vehs[appr] = cars[appr]+ pce['Two-wheeler']*two_wheelers[appr]+ pce['Three-wheeler']*three_wheelers[appr]+ pce['Heavy-vehicles']*heavy_vehicles[appr]
                        
        
    # %% Signal timings
    st.write("________________________________________________________________________________________")
    st.subheader("Signal Timing Input")
    l = int(st.slider('What is the lost time per phase', 0, 10, 4, key = 'lost_time'))
    st.write('Enter the effective green times of each approach')
    greens_columns = st.columns(n_appr)
    a = {} 
    a['v'] = list(vehs.values())
    a['s'] = list(s.values())
    a['n'] = list(n_virtual.values())
    demand = pd.DataFrame(a)
    demand['v_s'] = demand.v / demand.s
    greens0 = webster_timings(demand)
    bounds = []
    for appr in range(1, n_appr+1):
        with greens_columns[appr-1]:
            greens[appr] = float(st.text_input(f'Approach {appr}: ', 
                                           int(greens0[appr-1]), key = 'greens_'+str(appr)))
        bounds.append([8,201])
    bounds = tuple(bounds)  
    eff_green = sum(greens.values())
    C = eff_green + l*n_appr
    st.write("______________________________________________________________________________________")
    button_columns = st.columns(3)
    with button_columns[1]:
        calc_bool = st.button('Calculate Delays')
    with button_columns[2]:
        opt_bool = st.button('Optimise Signal Timings')
        
# %% DELAY CALCULATION
with st.expander('Delay calculation', calc_bool):
    st.header("Delay Calculation")
    tot_delay = []
    st.write("Individual approach specific and intersection control delays can be calculated in seconds/PCE assuming a split phasing")
    for appr in range(1, n_appr+1):
        delays[appr] = approach_delay(C,greens[appr],s[appr],vehs[appr],n_virtual[appr])
        if delays[appr] <= 0:
            tot_delay.append(np.inf)
        else:
            tot_delay.append(delays[appr]*vehs[appr])
    approaches = delays.keys()
    d = delays.values()
    delays_df = pd.DataFrame(list(zip(approaches, d, tot_delay)),
               columns =['Intersection Approach', 'Control delay (s/PCE)', 'Total Control delay (s)'])
    intersection_delay= sum(tot_delay)/sum(vehs.values())
    
    c1, c2 = st.columns([4,3])
    with c1:
        fig = px.bar(delays_df, x= 'Intersection Approach' , y = 'Control delay (s/PCE)')
        fig.update_layout(template = 'plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.subheader('Approach-Wise control delays')
        tab2 = go.Figure(data=[go.Table(
            header = dict(values = list(delays_df.columns),
                          align = 'center'),
            cells = dict(values =[delays_df['Intersection Approach'], delays_df['Control delay (s/PCE)'], delays_df['Total Control delay (s)']],
                         align = 'center'))
            ])
        tab2.update_layout(
            template = 'plotly_dark')
        #delays_df = delays_df.set_index('Intersection Approach')
        st.table(delays_df)
        #st.plotly_chart(tab2, use_container_width=False)
        st.subheader('Intersection Delay')
        st.metric(label="Average Intersection Control Delay", value=str(round(intersection_delay, 1))+" s/PCE")
    
# %% Signal Optimisation
with st.expander("Signal Optimisation", opt_bool):
    with st.container():
        st.header("Optimal Singal Timings")
        
        fun0 = obj_function(greens0)
        res = minimize(obj_function, greens0, method='trust-constr',
                       tol = 0.0001, bounds=bounds)    
        greens_opt = list(np.round(res.x, 1))
        eff_green_opt = sum(greens_opt)
        C_opt = eff_green_opt + l*n_appr
        inter_delay_after = (np.round(res.fun,1))
        inter_delay_before = (np.round(fun0,1))
        red_perc = round(100*(inter_delay_before - inter_delay_after)/inter_delay_before,1)
        
        delays_before = []
        delays_after = []
        
        for appr in range(1, n_appr+1):
            delays_before.append(approach_delay(C,greens0[appr-1],s[appr],vehs[appr],n_virtual[appr]))
            delays_after.append(approach_delay(C_opt,greens_opt[appr-1],s[appr],vehs[appr],n_virtual[appr]))
        
        df = pd.DataFrame(list(zip(approaches, greens0, greens_opt, delays_before, delays_after)),
                   columns = ['Intersection Approach', 'Initial Green (s)', 'Optimal Green (s)', 'Initial Control Delay (s/PCE)', 'Optimal Control Delay (s/PCE)'])
        
        fig2 = go.Figure(data=[
            go.Bar(name='Initial Green Times', x=df['Intersection Approach'], y=df['Initial Green (s)']),
            go.Bar(name='Optimal Green Times', x=df['Intersection Approach'], y=df['Optimal Green (s)'])
        ])

        fig2.update_layout(barmode='group',
                           xaxis_title='Intersection Approach',
                           yaxis_title='Gren time (seconds)',
                           template = 'plotly_dark')
        st.subheader("Comparison of the sighnal timings")
        st.plotly_chart(fig2, use_container_width=True)
        c_cols = st.columns([4,3,4])
        c_cols[1].metric("Optimal Cycle Time", str(round(sum(greens_opt))+l*n_appr)+" seconds", str(round(sum(greens_opt)) - round(sum(greens0)))+' seconds')

        
    with st.container():
        fig3 = go.Figure()
        
        fig3.add_trace(go.Scatterpolar(
              r=df['Initial Control Delay (s/PCE)'],
              theta=list(map(str, list(df['Intersection Approach'].values))),
              fill='toself',
              name='Initial Approach Delays'
        ))
        fig3.add_trace(go.Scatterpolar(
              r=df["Optimal Control Delay (s/PCE)"],
              theta=list(map(str, list(df['Intersection Approach'].values))),
              fill='toself',
              name='Optimal Approach Delays'
        ))
        
        fig3.update_layout(
            template = 'plotly_dark',
            polar = dict(
                radialaxis=dict(
                    title = 'Control Delay (PCE/s)',
                    showgrid = True
                    )
                )
            )
    
        st.write("________________________________________________________________________________________")
        st.subheader("Comparison of Approach Delays")
        st.plotly_chart(fig3, use_container_width=True)

    with st.container():
        cl1, cl2, cl3 = st.columns(3)
        cl1.metric("Initial Intersection Delay", str(round(inter_delay_before, 1))+" s/PCE")
        cl2.metric("Optimal Intersection Delay", str(round(inter_delay_after, 1))+" s/PCE")
        cl3.metric("Reduction in Intersection Delay", str(red_perc)+" %")
                
  
