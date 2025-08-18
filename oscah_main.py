# -*- coding-utf-8 -*-
"""
Created on Wed Apr 30 08:50:39 2025
Modified on Mon Aug 18 08:10:20 2025

@author: seyedhyd 
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math
import json
from io import StringIO

# %% App Configuration and Initialization
st.set_page_config(
    page_title="OSCAH Pro - Signal Analysis",
    page_icon="🚦",
    layout="wide"
)

# %% Core Calculation Functions
def approach_delay(C, g, s, v, n):
    """Calculates the control delay for a single intersection approach."""
    if s == 0 or C == 0 or g <= 0: return np.inf
    lam = g / C
    X = v / (s * lam) if (s * lam) > 0 else np.inf
    if X >= 1: return np.inf
    d1 = 0.5 * C * ((1 - lam)**2) / (1 - v/s)
    d2_term = 2 * n * lam * (1 - X)
    d2 = (X**(math.sqrt(2 * (n + 1)))) / d2_term if d2_term > 0 else np.inf
    return d1 + d2 + 4.84*lam - 13.15

def get_los(delay):
    """Maps control delay to Level of Service (LOS) grade."""
    if delay <= 10: return 'A'
    if delay <= 20: return 'B'
    if delay <= 35: return 'C'
    if delay <= 55: return 'D'
    if delay <= 80: return 'E'
    return 'F'

@st.cache_data
def webster_initial_timings(demand_df, l):
    """Calculates initial green times using Webster's method."""
    Y = demand_df['v_s'].sum()
    if Y >= 1: return [10] * len(demand_df)
    L = len(demand_df) * l
    C = math.ceil((1.5 * L + 5) / (1 - Y))
    eff_g = C - L
    greens = [round((eff_g / Y) * row.v_s) if Y > 0 else 10 for _, row in demand_df.iterrows()]
    return [max(min(int(g), 200), 8) for g in greens]

def oscah_cycle_length(sat_flow_rate, Y, L):
    """Calculates cycle length using the custom OSCAH formula."""
    if Y >= 1: return 240
    if 2000 <= sat_flow_rate < 2500:
        return math.ceil((1.72*L + 2.4)/(1-Y)) if Y <= 0.7 else math.ceil((1.65*L + 2.61)/(1-Y))
    elif 2500 <= sat_flow_rate <= 3000:
        return math.ceil((1.61*L + 2.27)/(1-Y)) if Y <= 0.7 else math.ceil((1.55*L + 2.31)/(1-Y))
    else:
        return math.ceil((1.5*L + 5)/(1-Y))

# %% Session State Management
def initialize_state():
    """Initializes session state with default values if they don't exist."""
    defaults = {
        'mode': 'Analyze Existing Timings', 'n_appr': 4, 'sat_flow_rate': 2900, 'lost_time': 4,
        'pce_enter': False, 'pce_tw': 0.78, 'pce_thw': 1.92, 'pce_hv': 3.42,
        'n_virt_enter': False, 'demand_pce_enter': True, 'results': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Per-approach defaults
    for i in range(1, 9):
        if f'n_actual_{i}' not in st.session_state: st.session_state[f'n_actual_{i}'] = 3
        if f'n_virtual_{i}' not in st.session_state: st.session_state[f'n_virtual_{i}'] = 5
        if f'vehs_pce_{i}' not in st.session_state: st.session_state[f'vehs_pce_{i}'] = {1: 1900, 2: 800, 3: 1500, 4: 900}.get(i, 1000)
        if f'cars_{i}' not in st.session_state: st.session_state[f'cars_{i}'] = {1: 900, 2: 350, 3: 700, 4: 400}.get(i, 500)
        if f'tw_{i}' not in st.session_state: st.session_state[f'tw_{i}'] = {1: 950, 2: 450, 3: 800, 4: 500}.get(i, 500)
        if f'thw_{i}' not in st.session_state: st.session_state[f'thw_{i}'] = {1: 100, 2: 34, 3: 65, 4: 36}.get(i, 50)
        if f'hv_{i}' not in st.session_state: st.session_state[f'hv_{i}'] = {1: 20, 2: 10, 3: 15, 4: 12}.get(i, 10)
        if f'greens_current_{i}' not in st.session_state: st.session_state[f'greens_current_{i}'] = 30
        if f'greens_baseline_{i}' not in st.session_state: st.session_state[f'greens_baseline_{i}'] = 30

initialize_state()

# %% Sidebar for User Inputs
with st.sidebar:
    st.title("🚦 OSCAH Pro")
    st.header("Configuration")

    with st.expander("📁 Scenario Management", expanded=True):
        uploaded_file = st.file_uploader("Load Scenario", type=['json'], help="Upload a JSON file to restore all inputs.")
        if uploaded_file:
            try:
                config = json.load(StringIO(uploaded_file.getvalue().decode("utf-8")))
                for key, value in config.items(): st.session_state[key] = value
                st.success("Scenario loaded!")
            except Exception as e: st.error(f"Error loading file: {e}")

        config_to_save = {k: v for k, v in st.session_state.items() if k != 'results'}
        st.download_button("Save Scenario", json.dumps(config_to_save, indent=2), "oscah_scenario.json", "application/json", help="Save all current inputs to a JSON file.")

    st.radio("Select Goal:", ('Analyze Existing Timings', 'Design & Compare Timings'), key='mode', help="**Analyze**: Evaluate known signal timings. **Design**: Create new timings and compare to a baseline.")
    st.markdown("---")

    st.header("Field Calibration")
    st.number_input('Number of approaches', min_value=2, max_value=8, key='n_appr', help="Total number of signalized approaches.")
    st.number_input("Saturation flow rate (PCE/hr/lane)", 1500, 4000, key='sat_flow_rate', step=50, help="Maximum flow rate if the signal were always green.")
    st.slider('Lost time per phase (s)', 0, 10, key='lost_time', help="Time during which the intersection is not used by any approach.")
    
    st.checkbox('Manually enter virtual lanes?', key='n_virt_enter', help="Check this to specify the number of parallel movements (virtual lanes) per approach.")
    if st.session_state.n_virt_enter:
        st.write("**Virtual Lanes**")
        v_lanes_cols = st.columns(st.session_state.n_appr)
        for i in range(1, st.session_state.n_appr + 1):
            with v_lanes_cols[i-1]: st.number_input(f'Appr. {i}', 1, 15, key=f'n_virtual_{i}')

    st.checkbox('Enter custom PCE values?', key='pce_enter', help="Check this to override the default Passenger Car Equivalent values.")
    if st.session_state.pce_enter:
        st.write("**PCE Values**")
        pce_cols = st.columns(3)
        pce_cols[0].number_input("Two-Wheeler", 0.1, 2.0, key='pce_tw', step=0.01)
        pce_cols[1].number_input("Three-Wheeler", 0.5, 4.0, key='pce_thw', step=0.01)
        pce_cols[2].number_input("Heavy Vehicle", 1.0, 6.0, key='pce_hv', step=0.01)
    st.markdown("---")

    st.header("Traffic & Signal Data")
    st.write("**Number of Physical Lanes**")
    lanes_cols = st.columns(st.session_state.n_appr)
    for i in range(1, st.session_state.n_appr + 1):
        with lanes_cols[i-1]: st.number_input(f'Appr. {i}', 1, 10, key=f'n_actual_{i}')

    st.checkbox('Enter demand in total PCE?', key='demand_pce_enter', help="Check to input one total PCE value per approach. Uncheck to enter counts by vehicle class.")
    if st.session_state.demand_pce_enter:
        st.write("**Traffic Demand (Total PCE/hr)**")
        demand_cols = st.columns(st.session_state.n_appr)
        for i in range(1, st.session_state.n_appr + 1):
            with demand_cols[i-1]: st.number_input(f'Appr. {i}', 0, 10000, key=f'vehs_pce_{i}')
    else:
        st.write("**Traffic Demand (Vehicles/hr by Class)**")
        for i in range(1, st.session_state.n_appr + 1):
            with st.expander(f"Approach {i} Demand"):
                v_cols = st.columns(4)
                v_cols[0].number_input("Cars", 0, 5000, key=f'cars_{i}')
                v_cols[1].number_input("Two-wheelers", 0, 5000, key=f'tw_{i}')
                v_cols[2].number_input("Three-wheelers", 0, 5000, key=f'thw_{i}')
                v_cols[3].number_input("Heavy Vehicles", 0, 5000, key=f'hv_{i}')

    st.write("**Green Times (seconds)**")
    greens_cols = st.columns(st.session_state.n_appr)
    if st.session_state.mode == 'Analyze Existing Timings':
        st.caption("Enter the current, known green times.")
        for i in range(1, st.session_state.n_appr + 1):
            with greens_cols[i-1]: st.number_input(f'Appr. {i}', 8, 200, key=f'greens_current_{i}')
    else: 
        st.caption("Defaults are from Webster's. Adjust for your baseline.")
        s_temp = {i: st.session_state.sat_flow_rate * st.session_state[f'n_actual_{i}'] for i in range(1, st.session_state.n_appr + 1)}
        v_temp = {i: st.session_state[f'vehs_pce_{i}'] for i in range(1, st.session_state.n_appr + 1)}
        df_temp = pd.DataFrame({'v': v_temp.values(), 's': s_temp.values()}); df_temp['v_s'] = df_temp.v / df_temp.s
        greens0 = webster_initial_timings(df_temp, st.session_state.lost_time)
        for i in range(1, st.session_state.n_appr + 1):
            with greens_cols[i-1]: st.number_input(f'Appr. {i}', 8, 200, value=greens0[i-1], key=f'greens_baseline_{i}')

# %% Main Panel with Tabs
tab1, tab2, tab3 = st.tabs(["📊 Analysis & Design", "📈 Sensitivity Analysis", "📖 Documentation"])

with tab1:
    st.header(f"🚦 OSCAH: {st.session_state.mode}")
    button_text = 'Analyze Performance' if st.session_state.mode == 'Analyze Existing Timings' else 'Design and Compare Timings'
    
    if st.button(button_text, type="primary", use_container_width=True):
        n_appr = st.session_state.n_appr
        s = {i: st.session_state.sat_flow_rate * st.session_state[f'n_actual_{i}'] for i in range(1, n_appr + 1)}
        n_v = {i: st.session_state[f'n_virtual_{i}'] if st.session_state.n_virt_enter else st.session_state[f'n_actual_{i}'] + 2 for i in range(1, n_appr + 1)}
        v = {}
        if st.session_state.demand_pce_enter:
            for i in range(1, n_appr + 1): v[i] = st.session_state[f'vehs_pce_{i}']
        else:
            pce = {'tw': st.session_state.pce_tw, 'thw': st.session_state.pce_thw, 'hv': st.session_state.pce_hv}
            for i in range(1, n_appr + 1):
                v[i] = (st.session_state[f'cars_{i}'] + st.session_state[f'tw_{i}'] * pce['tw'] +
                        st.session_state[f'thw_{i}'] * pce['thw'] + st.session_state[f'hv_{i}'] * pce['hv'])
        greens_key_prefix = 'greens_current_' if st.session_state.mode == 'Analyze Existing Timings' else 'greens_baseline_'
        greens = {i: st.session_state[f'{greens_key_prefix}{i}'] for i in range(1, n_appr + 1)}
        Y_check_df = pd.DataFrame({'v': v.values(), 's': s.values()}); Y_check_df['v_s'] = Y_check_df.v / Y_check_df.s
        if Y_check_df['v_s'].sum() >= 1.0:
            st.error("🚨 **Intersection Over Capacity!** The total demand exceeds capacity (Σv/s ≥ 1). Reduce demand or increase lanes.", icon="🔥")
            st.session_state.results = None
        else:
            if st.session_state.mode == 'Analyze Existing Timings':
                C = sum(greens.values()) + (n_appr * st.session_state.lost_time)
                delays = [approach_delay(C, greens[i], s[i], v[i], n_v[i]) for i in range(1, n_appr + 1)]
                st.session_state.results = {'type': 'analysis', 'C': C, 'delays': delays, 'greens': greens, 'v':v}
            else:
                C_base = sum(greens.values()) + (n_appr * st.session_state.lost_time)
                delays_base = [approach_delay(C_base, greens[i], s[i], v[i], n_v[i]) for i in range(1, n_appr + 1)]
                Y = Y_check_df['v_s'].sum()
                L = n_appr * st.session_state.lost_time
                C_osc = oscah_cycle_length(st.session_state.sat_flow_rate, Y, L)
                eff_g_osc = C_osc - L
                greens_oscah = [round((eff_g_osc / Y) * vs) if Y > 0 else 10 for vs in Y_check_df['v_s']]
                greens_oscah = [max(min(g, 200), 8) for g in greens_oscah]
                delays_os = [approach_delay(C_osc, greens_oscah[i-1], s[i], v[i], n_v[i]) for i in range(1, n_appr + 1)]
                st.session_state.results = {'type': 'design', 'C_base': C_base, 'delays_base': delays_base, 'greens_base': greens,
                                            'C_osc': C_osc, 'delays_os': delays_os, 'greens_oscah': greens_oscah, 'v':v}

    if st.session_state.results:
        results = st.session_state.results
        n_appr = len(results['v'])
        approaches = list(range(1, n_appr + 1))
        
        if results['type'] == 'analysis':
            avg_delay = np.average(results['delays'], weights=list(results['v'].values()))
            st.subheader("Performance Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Cycle Length", f"{results['C']:.0f} s")
            col2.metric("Avg. Intersection Delay", f"{avg_delay:.1f} s/PCE")
            col3.metric("Intersection LOS", get_los(avg_delay))
            st.markdown("---")
            df = pd.DataFrame({'Approach': approaches, 'Green Time (s)': list(results['greens'].values()), 'Delay (s/PCE)': results['delays']})
            df['LOS'] = df['Delay (s/PCE)'].apply(get_los)
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Results per Approach")
                st.dataframe(df.set_index('Approach'), use_container_width=True)
            with c2:
                st.subheader("Delay Visualization")
                fig = px.bar(df, x='Approach', y='Delay (s/PCE)', template='plotly_dark', title="Delay per Approach")
                st.plotly_chart(fig, use_container_width=True)

        elif results['type'] == 'design':
            avg_delay_base = np.average(results['delays_base'], weights=list(results['v'].values()))
            avg_delay_os = np.average(results['delays_os'], weights=list(results['v'].values()))
            st.subheader("Performance Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Baseline Performance")
                c1, c2, c3 = st.columns(3)
                c1.metric("Cycle", f"{results['C_base']:.0f} s"); c2.metric("Avg. Delay", f"{avg_delay_base:.1f} s/PCE"); c3.metric("LOS", get_los(avg_delay_base))
            with col2:
                st.markdown("##### OSCAH Design Performance")
                c1, c2, c3 = st.columns(3)
                c1.metric("Cycle", f"{results['C_osc']:.0f} s"); c2.metric("Avg. Delay", f"{avg_delay_os:.1f} s/PCE", delta=f"{avg_delay_base - avg_delay_os:.1f} s Improvement", delta_color="inverse"); c3.metric("LOS", get_los(avg_delay_os))
            st.markdown("---")
            
            df_delays = pd.DataFrame({'Approach': approaches, 'Baseline': results['delays_base'], 'OSCAH': results['delays_os']})
            c1, c2 = st.columns([3,2])
            with c1:
                st.subheader("Approach Delay Comparison")
                fig_bar = px.bar(df_delays, x='Approach', y=['Baseline', 'OSCAH'], barmode='group', template='plotly_dark', labels={'value': 'Delay (s/PCE)', 'variable': 'Method'})
                st.plotly_chart(fig_bar, use_container_width=True)
            with c2:
                st.subheader("Delay Distribution Radar Chart")
                fig_radar = go.Figure()
                theta_labels = [f"Approach {i}" for i in df_delays['Approach']]
                fig_radar.add_trace(go.Scatterpolar(r=df_delays['Baseline'], theta=theta_labels, fill='toself', name='Baseline'))
                fig_radar.add_trace(go.Scatterpolar(r=df_delays['OSCAH'], theta=theta_labels, fill='toself', name='OSCAH'))
                fig_radar.update_layout(template='plotly_dark', polar=dict(radialaxis=dict(visible=True, range=[0, max(df_delays['Baseline'].max(), df_delays['OSCAH'].max())*1.1])))
                st.plotly_chart(fig_radar, use_container_width=True)

        with st.expander("📄 View and Download Summary Report"):
            report_text = f"# OSCAH Signal Analysis Report\n\n## Mode: {st.session_state.mode}\n\n### Key Inputs\n- Number of Approaches: {n_appr}\n- Saturation Flow Rate: {st.session_state.sat_flow_rate} PCE/hr/lane\n- Lost Time per Phase: {st.session_state.lost_time} s\n\n"
            if results['type'] == 'analysis':
                avg_delay = np.average(results['delays'], weights=list(results['v'].values()))
                report_text += f"### Analysis Results\n- Cycle Length: {results['C']:.0f} s\n- Average Intersection Delay: {avg_delay:.1f} s/PCE\n- Intersection LOS: {get_los(avg_delay)}\n"
            else:
                avg_delay_os = np.average(results['delays_os'], weights=list(results['v'].values()))
                report_text += f"### Design Recommendations (OSCAH)\n- Recommended Cycle Length: {results['C_osc']:.0f} s\n- Predicted Average Delay: {avg_delay_os:.1f} s/PCE\n- Predicted Intersection LOS: {get_los(avg_delay_os)}\n\n| Approach | Recommended Green (s) |\n|---|---|\n"
                for i, g in enumerate(results['greens_oscah']): report_text += f"| {i+1} | {g} |\n"
            st.markdown(report_text)
            st.download_button("Download Report", report_text, "OSCAH_report.md")
    else:
        st.info("Configure your parameters in the sidebar and click the button above to begin. ⬆️")

with tab2:
    st.header("Sensitivity Analysis")
    st.write("Analyze how the intersection's average delay changes as you vary a single input parameter.")
    param_options = {f'Demand on Approach {i}': f'vehs_pce_{i}' for i in range(1, st.session_state.n_appr + 1)}
    param_options['Saturation Flow Rate'] = 'sat_flow_rate'
    selected_param_label = st.selectbox("Parameter to Vary", options=param_options.keys())
    selected_param_key = param_options[selected_param_label]
    current_val = st.session_state.get(selected_param_key, 1000)
    col1, col2, col3 = st.columns(3)
    min_val = col1.number_input("Minimum Value", value=int(current_val * 0.5), key='sens_min')
    max_val = col2.number_input("Maximum Value", value=int(current_val * 1.5), key='sens_max')
    steps = col3.number_input("Number of Steps", value=10, min_value=2, max_value=50, key='sens_steps')

    if st.button("Run Sensitivity Analysis", type="primary", use_container_width=True, key='sens_run'):
        if min_val >= max_val:
            st.error("Minimum value must be less than maximum value.")
        else:
            param_range = np.linspace(min_val, max_val, steps)
            results_delays = []
            with st.spinner("Running simulations..."):
                for val in param_range:
                    temp_state = dict(st.session_state)
                    temp_state[selected_param_key] = val
                    
                    n_appr = temp_state['n_appr']
                    s = {i: temp_state['sat_flow_rate'] * temp_state[f'n_actual_{i}'] for i in range(1, n_appr + 1)}
                    v = {i: temp_state[f'vehs_pce_{i}'] for i in range(1, n_appr + 1)}
                    n_v = {i: temp_state[f'n_virtual_{i}'] if temp_state['n_virt_enter'] else temp_state[f'n_actual_{i}'] + 2 for i in range(1, n_appr + 1)}
                    Y_df = pd.DataFrame({'v': v.values(), 's': s.values()}); Y_df['v_s'] = Y_df.v / Y_df.s
                    Y = Y_df['v_s'].sum()
                    if Y >= 1: avg_delay = np.inf
                    else:
                        L = n_appr * temp_state['lost_time']
                        C_osc = oscah_cycle_length(temp_state['sat_flow_rate'], Y, L)
                        eff_g = C_osc - L
                        greens = [round((eff_g / Y) * vs) if Y > 0 else 10 for vs in Y_df['v_s']]
                        greens = [max(min(g, 200), 8) for g in greens]
                        delays = [approach_delay(C_osc, greens[i-1], s[i], v[i], n_v[i]) for i in range(1, n_appr + 1)]
                        avg_delay = np.average(delays, weights=list(v.values())) if any(wt > 0 for wt in v.values()) else 0
                    results_delays.append(avg_delay)
            fig_df = pd.DataFrame({'Parameter Value': param_range, 'Average Delay (s/PCE)': results_delays})
            fig = px.line(fig_df, x='Parameter Value', y='Average Delay (s/PCE)', title=f'Sensitivity of Delay to {selected_param_label}', template='plotly_dark', markers=True)
            fig.update_layout(xaxis_title=selected_param_label)
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("📖 OSCAH Pro - User Guide & Technical Documentation")
    st.markdown("**Version 1.0** | **Last Updated:** August 18, 2025")
    st.markdown("---")

    with st.expander("1. Introduction", expanded=True):
        st.subheader("What is OSCAH Pro?")
        st.write("""
        **OSCAH Pro** (Optimal Signal Control and Analysis for Heterogeneous and Lane-Free Traffic) is a web-based tool designed for traffic engineers, planners, and students. It provides a powerful platform to analyze the performance of existing traffic signal timings and to design new, optimized signal plans for isolated intersections.
        The tool is specifically calibrated for **heterogeneous and lane-free traffic conditions**, which are common in many regions where vehicle fleets are diverse (including high numbers of two- and three-wheelers) and lane discipline is not strictly followed.
        """)
        st.subheader("Who is this for?")
        st.markdown("""
        - **Traffic Engineers & Planners:** Quickly evaluate intersection performance and design signal timings using a specialized model.
        - **Transportation Students & Researchers:** Understand the impact of different parameters on intersection delay and compare traffic flow models.
        - **Consultants:** Prepare data-driven reports on intersection performance and proposed improvements.
        """)

    with st.expander("2. Core Concepts & Formulas", expanded=True):
        st.subheader("Key Terminology")
        st.markdown("""
        - **Saturation Flow Rate (s):** The maximum number of vehicles (in PCE) that can pass through an intersection approach per hour if the signal light were green for the entire hour. It represents the capacity of the approach.
        - **Passenger Car Equivalent (PCE):** A factor used to convert different vehicle types into a standard unit.
        - **Virtual Lanes (n):** A concept to model lane-free traffic. In such conditions, vehicles often form more parallel queues than the number of physical lanes.
        - **Lost Time (l):** The time during each signal cycle when the intersection is not effectively used by any approach.
        - **Level of Service (LOS):** A standardized letter grade (A-F) that describes the operating conditions of an intersection based on the average control delay experienced by drivers.
        """)
        st.subheader("Key Formulas")
        
        st.markdown("**Approach Delay Model**")
        st.write("The total delay ($d$) for an approach is the sum of three components: uniform delay ($d_1$), random/overflow delay ($d_2$), and an empirical adjustment term ($d_3$).")
        st.latex(r'''
        d = d_1 + d_2 + d_3
        ''')
        st.markdown(r"""
        - **Uniform Delay ($d_1$):** Assumes vehicles arrive at a uniform rate and accounts for the delay when a queue clears after the signal turns green.
        """)
        st.latex(r'''
        d_1 = 0.5 \cdot C \cdot \frac{(1 - \lambda)^2}{1 - (v/s)}
        ''')
        st.markdown(r"""
        - **Random Delay ($d_2$):** Accounts for the additional delay caused by random fluctuations in vehicle arrivals from one cycle to the next.
        """)
        st.latex(r'''
        d_2 = \frac{X^{\sqrt{2(n+1)}}}{2n\lambda(1-X)}
        ''')
        st.markdown(r"""
        - **Empirical Adjustment ($d_3$):** A calibration term to better match field-observed conditions.
        """)
        st.latex(r'''
        d_3 = 4.84\lambda - 13.15
        ''')
        st.markdown(r"""
        *Where: $C$ = Cycle Length, $\lambda$ = Green Ratio ($g/C$), $v$ = Traffic Volume, $s$ = Saturation Flow, $X$ = Degree of Saturation, $n$ = Virtual lanes.*
        """)

        st.markdown("**OSCAH Cycle Length Model ($C_{oscah}$)**")
        st.write("The OSCAH model is an empirical formula that adjusts the cycle length based on the intersection's saturation flow rate and overall traffic intensity ($Y$). It uses different coefficients for different operational regimes.")
        st.latex(r'''
        C_{oscah} = \begin{cases} \lceil \frac{1.72 L + 2.4}{1 - Y} \rceil & \text{if } 2000 \le s < 2500 \text{ and } Y \le 0.7 \\ \lceil \frac{1.65 L + 2.61}{1 - Y} \rceil & \text{if } 2000 \le s < 2500 \text{ and } Y > 0.7 \\ \lceil \frac{1.61 L + 2.27}{1 - Y} \rceil & \text{if } 2500 \le s \le 3000 \text{ and } Y \le 0.7 \\ \lceil \frac{1.55 L + 2.31}{1 - Y} \rceil & \text{if } 2500 \le s \le 3000 \text{ and } Y > 0.7 \\ \lceil \frac{1.5 L + 5}{1 - Y} \rceil & \text{otherwise (Fallback to Webster)} \end{cases}
        ''')
        st.markdown(r"""
        *Where: $L$ = Total lost time for the intersection, $Y$ = Sum of critical flow ratios ($v/s$), $s$ = Saturation flow rate per lane.*
        """)

    with st.expander("3. User Guide: Step-by-Step"):
        st.markdown("""
        The OSCAH Pro interface is divided into the **sidebar** for configuration and the **main panel** for results.

        **Step 1: Scenario Management**
        - **Save Scenario:** After entering all your data, click this button to download a `.json` file containing all your input settings.
        - **Load Scenario:** Click the "Browse files" button to upload a previously saved `.json` file.

        **Step 2: Select Your Goal (Mode)**
        1.  **Analyze Existing Timings:** To evaluate the performance of known signal timings.
        2.  **Design & Compare Timings:** To calculate a new, optimized signal plan using the OSCAH model.

        **Step 3: Configure Intersection Parameters**
        Fill in all the required data in the sidebar, including Field Calibration, Traffic Data, and Green Times. Use the tooltips for help on each parameter.

        **Step 4: Run the Analysis & Interpret Results**
        Click the main button (`Analyze Performance` or `Design and Compare Timings`) to run the calculations. The results, including metrics, tables, and charts, will appear in the "Analysis & Design" tab.

        **Step 5 (Optional): Run a Sensitivity Analysis**
        Navigate to the "Sensitivity Analysis" tab to explore how changes in one variable affect intersection performance.
        """)

    with st.expander("4. Troubleshooting"):
        st.subheader("Error: 'Intersection Over Capacity!'")
        st.warning("""
        This is the most common error. It means that the traffic demand you have entered is greater than the physical capacity of the intersection (`Σv/s ≥ 1`). The delay in this scenario is theoretically infinite.
        
        **Solution:** You must either **reduce the traffic demand** values or **increase the capacity** by adding more lanes or increasing the saturation flow rate.
        """)
