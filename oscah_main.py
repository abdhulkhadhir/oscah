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
import plotly.graph_objects as go
from scipy.optimize import minimize, Bounds, brute

# %% Constants initialisation
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
delays = {}
s = {}

# %% Functions
def approach_delay(C, g, s, v, n):
    lam = g/C
    X = v/(s*lam)
    d1 = 0.5*C*(((1-lam)**2)/(1-v/s))
    d2 = (X**(math.sqrt(2*(n+1)))) / (2*n*lam*(1-X))
    if d2 < 0:
        d2 = np.inf
    return d1 + d2 + 4.84*lam - 13.15

def webster_timings(demand):
    Y = demand.v_s.sum()
    L = len(demand)*l
    C = math.ceil((1.5*L + 5) / (1-Y))
    eff_g = C - L
    demand['greens_actual'] = round((eff_g/Y) * demand.v_s)
    demand['greens_limitted'] = [max(min(x, 201), 8) for x in demand['greens_actual']]
    return list(demand.greens_limitted)

def oscah_cycle_length(sat_flow_rate, Y, L):
    if 2000 <= sat_flow_rate < 2500:
        return math.ceil((1.72*L + 2.4)/(1-Y)) if Y <= 0.7 else math.ceil((1.65*L + 2.61)/(1-Y))
    elif 2500 <= sat_flow_rate <= 3000:
        return math.ceil((1.61*L + 2.27)/(1-Y)) if Y <= 0.7 else math.ceil((1.55*L + 2.31)/(1-Y))
    else:
        return math.ceil((1.5*L + 5)/(1-Y))  # fallback

def obj_function(greens):
    avg_delay, tot_delay = [], []
    C = sum(greens) + len(demand)*l
    for i in range(len(demand)):
        d_i = approach_delay(C, greens[i], demand.s[i], demand.v[i], demand.n[i])
        if d_i <= 0:
            avg_delay.append(np.inf); tot_delay.append(np.inf)
        else:
            avg_delay.append(d_i); tot_delay.append(d_i * demand.v[i])
    demand['avg_delay'], demand['tot_delay'] = avg_delay, tot_delay
    return demand.tot_delay.sum()/demand.v.sum()

#%% App layout and inputs
st.title('Optimal Signal Control and Analysis for Heterogeneous and Lane-Free Traffic (OSCAH)')

with st.expander('Input Parameters'):
    st.header('Field Calibration Constants')
    col1, col2 = st.columns(2)
    with col1:
        sat_flow_rate = float(st.text_input("Enter saturation flow rate in PCE/hr/lane (if known)", 2900))
    with col2:
        n_appr = int(st.text_input('Enter the number of intersection approaches', 4))

    # Actual lanes
    st.write("________________________________________________________________________________________")
    st.write("Enter the number of lanes in each Approach")
    lanes_columns = st.columns(n_appr)
    for appr in range(1, n_appr+1):
        with lanes_columns[appr-1]:
            n_actual[appr] = int(st.text_input(f'Approach {appr}: ', 3, key=f'n_actual_{appr}'))
            s[appr] = sat_flow_rate * n_actual[appr]

    # Virtual lanes
    st.write("________________________________________________________________________________________")
    st.write("Do you know the number of parallel movmeents / virtual lanes present in each approach?")
    n_virt_enter = st.checkbox('Number of parallel movements / virtual lanes known?')
    if n_virt_enter:
        virtual_columns = st.columns(n_appr)
        for appr in range(1, n_appr+1):
            with virtual_columns[appr-1]:
                n_virtual[appr] = int(st.text_input(f'Approach {appr}: ', 5, key=f'n_virtual_{appr}'))
    else:
        for appr in range(1, n_appr+1):
            n_virtual[appr] = n_actual[appr] + 2

    # PCE values
    pce = {'Two-wheeler':0.78,'Three-wheeler':1.92,'Heavy-vehicles':3.42}
    st.write("________________________________________________________________________________________")
    st.write("Do you know the Passenger Car Equivalent Values of all the vehicle types?")
    st.subheader('Default PCE values')
    pc_cols = st.columns(4)
    pc_cols[0].metric("Car",1.00); pc_cols[1].metric("Two-Wheeler",0.78)
    pc_cols[2].metric("Three-Wheeler",1.92); pc_cols[3].metric("Heavy Vehicle",3.42)
    pce_enter = st.checkbox('PCE values known?')
    if pce_enter:
        st.write("Enter desired PCE values")
        p_cols = st.columns(3)
        for i,(veh,val) in enumerate(pce.items()):
            with p_cols[i]:
                pce_used[veh] = float(st.text_input(f'{veh}: ', val, key=f'pce_user_{veh}'))
    else:
        pce_used = pce

    # Traffic demand
    st.write("________________________________________________________________________________________")
    st.header('Traffic Demand and Signal Timiings')
    st.subheader("Traffic Demand Inputs")
    demand_pce_enter = st.checkbox('Demand values in PCE known?', True)
    if demand_pce_enter:
        cols = st.columns(n_appr)
        for appr in range(1, n_appr+1):
            with cols[appr-1]:
                vehs[appr] = float(st.text_input(f'Approach {appr}: ', vehs_pce_default[appr], key=f'vehs_pce_{appr}'))
    else:
        for appr in range(1, n_appr+1):
            st.write(f"Enter number of vehicles in each class for approach {appr}")
            cls_cols = st.columns(4)
            with cls_cols[0]:
                cars[appr] = int(st.text_input('Cars: ', cars_default[appr], key=f'cars_{appr}'))
            with cls_cols[1]:
                two_wheelers[appr] = int(st.text_input('Two wheelers: ', two_wheelers_default[appr], key=f'tw_{appr}'))
            with cls_cols[2]:
                three_wheelers[appr] = int(st.text_input('Three wheelers: ', three_wheelers_default[appr], key=f'thw_{appr}'))
            with cls_cols[3]:
                heavy_vehicles[appr] = int(st.text_input('Heavy vehicles: ', heavy_vehicles_default[appr], key=f'hv_{appr}'))
            vehs[appr] = (cars[appr]
                          + pce['Two-wheeler']*two_wheelers[appr]
                          + pce['Three-wheeler']*three_wheelers[appr]
                          + pce['Heavy-vehicles']*heavy_vehicles[appr])

# %% Signal timings
st.write("________________________________________________________________________________________")
st.subheader("Signal Timing Input")
l = int(st.slider('What is the lost time per phase', 0, 10, 4, key='lost_time'))
st.write('Enter the effective green times of each approach')
greens_cols = st.columns(n_appr)

# prepare demand DataFrame
a = {'v':list(vehs.values()), 's':list(s.values()), 'n':list(n_virtual.values())}
demand = pd.DataFrame(a)
demand['v_s'] = demand.v / demand.s

# Webster initial greens
greens0 = webster_timings(demand)
for appr in range(1, n_appr+1):
    with greens_cols[appr-1]:
        greens[appr] = float(st.text_input(f'Approach {appr}: ',
                                           int(greens0[appr-1]),
                                           key=f'greens_{appr}'))

# bounds
bounds = tuple([[8,201] for _ in range(n_appr)])

# cycle times
Y = demand.v_s.sum()
L = len(demand) * l
C_web = sum(greens.values()) + L
C_osc = oscah_cycle_length(sat_flow_rate, Y, L)
eff_g_osc = C_osc - L
greens_oscah = list(round((eff_g_osc/Y) * demand.v_s))
greens_oscah = [max(min(x,201),8) for x in greens_oscah]

# optimization
fun0 = obj_function(greens0)
res = minimize(obj_function, greens0, method='trust-constr', tol=1e-4, bounds=bounds)
greens_opt = list(np.round(res.x))
C_opt = sum(greens_opt) + L

st.write("______________________________________________________________________________________")
btn1, btn2, btn3 = st.columns(3)
with btn2:
    calc_bool = st.button('Calculate Delays')
with btn3:
    opt_bool  = st.button('Optimise Signal Timings')

# %% DELAY CALCULATION
with st.expander('Delay calculation', calc_bool):
    st.header("Delay Calculation")

    # build per‐approach delay lists
    delays_web = []
    delays_os = []
    delays_op = []
    approaches = list(range(1, n_appr+1))
    for i in approaches:
        # Webster/Initial
        d_w = approach_delay(C_web, greens[i], s[i], vehs[i], n_virtual[i])
        delays_web.append(d_w)
        # OSCAH
        d_o = approach_delay(C_osc, greens_oscah[i-1], s[i], vehs[i], n_virtual[i])
        delays_os.append(d_o)
        # Optimized
        d_p = approach_delay(C_opt, greens_opt[i-1], s[i], vehs[i], n_virtual[i])
        delays_op.append(d_p)

    # now create DataFrame from equal‐length lists
    df_del = pd.DataFrame({
        'Intersection Approach': approaches,
        'Initial Delay (s/PCE)': delays_web,
        'OSCAH Delay (s/PCE)':   delays_os,
        'Optimized Delay (s/PCE)':delays_op
    })

    # intersection‐wide Webster delay metric
    inter_delay = (df_del['Initial Delay (s/PCE)'] * list(vehs.values())).sum() / sum(vehs.values())

    # bar chart
    c1, c2 = st.columns([4,3])
    with c1:
        fig = px.bar(df_del,
                     x='Intersection Approach',
                     y=['Initial Delay (s/PCE)', 'OSCAH Delay (s/PCE)', 'Optimized Delay (s/PCE)'])
        fig.update_layout(template='plotly_dark', barmode='group')
        st.plotly_chart(fig, use_container_width=True)

    # table + metric
    with c2:
        st.table(df_del.set_index('Intersection Approach'))
        st.subheader('Intersection Delay (Webster)')
        st.metric("Average Intersection Control Delay",
                  f"{round(inter_delay,1)} s/PCE")

# %% Signal Optimisation
with st.expander("Signal Optimisation", opt_bool):
    st.header("Optimal Signal Timings")
    
    # Compute intersection‐wide metrics
    inter_before = round(fun0, 1)
    inter_after  = round(res.fun, 1)
    red_perc     = round(100 * (inter_before - inter_after) / inter_before, 1)

    # Approach indices
    approaches = list(range(1, n_appr + 1))

    # Build the detailed table of green times
    df_cycle = pd.DataFrame({
        'Approach': approaches,
        'Webster Green (s)': [greens[i] for i in approaches],
        'OSCAH Green (s)':    greens_oscah,
        'Optimized Green (s)':greens_opt
    })

    # Add a “Cycle Length” row at the bottom
    cycle_row = pd.DataFrame({
        'Approach': ['Cycle Length'],
        'Webster Green (s)': [round(C_web)],
        'OSCAH Green (s)':    [round(C_osc)],
        'Optimized Green (s)': [round(C_opt)]
    })
    df_cycle = pd.concat([df_cycle, cycle_row], ignore_index=True)

    # Show cycle lengths as individual metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Webster Cycle", f"{round(C_web)} s")
    c2.metric("OSCAH Cycle",   f"{round(C_osc)} s")
    c3.metric("Optimised Cycle",f"{round(C_opt)} s")
    c4.metric("Delay Reduction", f"{red_perc} %")

    st.subheader("Green Times & Cycle Lengths Comparison")
    st.table(df_cycle.set_index('Approach'))

    # Radar plot for delays
    fig3 = go.Figure()
    theta = df_cycle.loc[:n_appr-1, 'Approach'].astype(str)
    fig3.add_trace(go.Scatterpolar(
        r=df_del['Initial Delay (s/PCE)'], theta=theta, fill='toself', name='Webster'))
    fig3.add_trace(go.Scatterpolar(
        r=df_del['OSCAH Delay (s/PCE)'], theta=theta, fill='toself', name='OSCAH'))
    fig3.add_trace(go.Scatterpolar(
        r=df_del['Optimized Delay (s/PCE)'], theta=theta, fill='toself', name='Optimized'))
    fig3.update_layout(
        template='plotly_dark',
        polar=dict(radialaxis=dict(title='Control Delay (s/PCE)'))
    )
    st.write("________________________________________________________________________________________")
    st.subheader("Comparison of Approach Delays")
    st.plotly_chart(fig3, use_container_width=True)

    # Intersection delay metrics
    cl1, cl2, cl3 = st.columns(3)
    cl1.metric("Webster Avg Delay", f"{round(inter_before,1)} s/PCE")
    cl2.metric("OSCAH Avg Delay",   f"{round((np.array(delays_os) * np.array(list(vehs.values()))).sum() / sum(vehs.values()),1)} s/PCE")
    cl3.metric("Optimised Avg Delay",f"{round(inter_after,1)} s/PCE")
