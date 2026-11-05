import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from itertools import product
from datetime import datetime

st.set_page_config(page_title="Intelligent Power Quality Monitor", page_icon="⚡", layout="wide")
st.title("⚡ Intelligent Single-Phase Power Quality Monitoring")
st.caption("Human-controlled loads • APFC • ML • Load shedding • Digital twin • Alerts • Energy")

with st.sidebar:
    st.header("Operator controls")
    voltage = st.slider("Supply voltage (V RMS)", 180, 250, 230)
    freq = st.select_slider("Frequency (Hz)", options=[45,48,49,50,51,52,55], value=50)
    fs = st.select_slider("Sampling rate (S/s)", options=[4000,8000,12000,16000], value=8000)
    window_ms = st.select_slider("Measurement window (ms)", options=[100,200,400], value=200)
    st.subheader("SCADA switching")
    breaker_closed = st.toggle("Main breaker — closed", True, key="main_breaker_closed")
    st.caption("Open the breaker to de-energize all downstream loads.")
    st.subheader("Manual load switches")
    sw_r = st.toggle("Resistive load — heater", True)
    sw_l = st.toggle("Inductive load — motor", False)
    sw_c = st.toggle("Capacitive load", False)
    sw_n = st.toggle("Nonlinear load — rectifier", False)
    st.divider()
    st.subheader("Automatic control")
    apfc = st.toggle("Enable automatic PF correction", False)
    target_pf = st.slider("Target displacement PF", .80, .99, .95, .01)
    shedding = st.toggle("Enable automatic load shedding", False)
    current_limit = st.slider("Current limit (A)", 1.0, 10.0, 4.5, .1)
    st.caption("Shedding priority: heater → nonlinear load → motor.")
    st.divider()
    st.subheader("Alerts and energy")
    tariff = st.number_input("Tariff (₹/kWh)", min_value=0.0, value=8.0, step=.5)
    v_low = st.number_input("Undervoltage threshold (V)", value=207.0)
    v_high = st.number_input("Overvoltage threshold (V)", value=253.0)
    thd_limit = st.number_input("Current THD alert (%)", value=10.0)

def rms(x):
    return float(np.sqrt(np.mean(x*x)))

def fundamental(x,t,f):
    w=2*np.pi*f*t
    a=2/len(x)*np.sum(x*np.cos(w))
    b=2/len(x)*np.sum(x*np.sin(w))
    return np.hypot(a,b)/np.sqrt(2), np.arctan2(a,b)

def simulate(loads, cap_uF=0, switching_event=False):
    n=int(fs*window_ms/1000)
    t=np.arange(n)/fs
    w=2*np.pi*freq
    vp=np.sqrt(2)*voltage
    nonlinear="Nonlinear" in loads
    v=vp*np.sin(w*t)+vp*(.008+(.01 if nonlinear else 0))*np.sin(3*w*t+.15)+vp*(.004+(.005 if nonlinear else 0))*np.sin(5*w*t-.2)
    i=np.zeros(n)
    if "Resistive" in loads: i+=np.sqrt(2)*2*np.sin(w*t)
    if "Inductive" in loads: i+=np.sqrt(2)*1.8*np.sin(w*t-np.deg2rad(48))
    if "Capacitive" in loads: i+=np.sqrt(2)*1.1*np.sin(w*t+np.deg2rad(38))
    if nonlinear:
        i+=np.sqrt(2)*1.35*np.sin(w*t-np.deg2rad(12))
        i+=np.sqrt(2)*.72*np.sin(3*w*t-np.deg2rad(35))
        i+=np.sqrt(2)*.38*np.sin(5*w*t-np.deg2rad(55))
        i+=np.sqrt(2)*.20*np.sin(7*w*t-np.deg2rad(70))
    if cap_uF:
        i+=np.sqrt(2)*voltage*(2*np.pi*freq)*cap_uF*1e-6*np.sin(w*t+np.pi/2)
    if switching_event:
        center=int(min(n-1, .035*fs))
        pulse_t=t-(center/fs)
        envelope=np.exp(-np.maximum(pulse_t,0)/.012)*(pulse_t>=0)
        ring=np.sin(2*np.pi*420*pulse_t)*envelope
        v+=vp*.16*ring
        i+=np.sqrt(2)*2.8*ring
    return t,v,i

def analyze(t,v,i):
    vr,ir=rms(v),rms(i)
    p=float(np.mean(v*i)); s=vr*ir
    v1,pv=fundamental(v,t,freq); i1,pi=fundamental(i,t,freq)
    angle=float(np.rad2deg(np.angle(np.exp(1j*(pv-pi)))))
    q1=float(v1*i1*np.sin(np.deg2rad(angle)))
    def fft_table(x):
        win=np.hanning(len(x)); cg=np.sum(win)/len(win)
        spec=np.fft.rfft((x-np.mean(x))*win)
        hz=np.fft.rfftfreq(len(x),1/fs)
        mag=2*np.abs(spec)/(len(x)*cg*np.sqrt(2))
        rows=[]
        for h in range(1,41):
            ix=int(np.argmin(abs(hz-h*freq)))
            rows.append((h,hz[ix],mag[ix]))
        return pd.DataFrame(rows,columns=["Order","Frequency (Hz)","RMS magnitude"])
    vh,ih=fft_table(v),fft_table(i)
    def thd(tab):
        a=tab["RMS magnitude"].to_numpy()
        return float(np.sqrt(np.sum(a[1:]**2))/a[0]*100) if a[0] else 0
    return {"vr":vr,"ir":ir,"p":p,"s":s,"pf":p/s if s else 0,
            "angle":angle,"q":q1,"vthd":thd(vh),"ithd":thd(ih),"vh":vh,"ih":ih}

commanded=[name for name,on in [("Resistive",sw_r),("Inductive",sw_l),("Capacitive",sw_c),("Nonlinear",sw_n)] if on]
switch_state=(breaker_closed,tuple(commanded),apfc)
switching_event=st.session_state.get("transient_switch_state") is not None and (
    switch_state != st.session_state.transient_switch_state)
st.session_state.transient_switch_state=switch_state
requested=commanded if breaker_closed else []
t,v,i=simulate(requested, switching_event=switching_event)
m=analyze(t,v,i)

# Discrete APFC stages: select combination minimizing residual fundamental Q, only for lagging loads.
stages=[2.,4.,8.]
cap=0.; chosen=[]
if apfc and m["q"]>1:
    q_target=m["p"]*np.tan(np.arccos(target_pf))
    best=abs(m["q"]-q_target)
    for bits in product([0,1],repeat=3):
        total=sum(b*x for b,x in zip(bits,stages))
        qcap=voltage**2*2*np.pi*freq*total*1e-6
        err=abs(m["q"]-qcap-q_target)
        if err<best: best,cap=err,total
    chosen=[x for x in stages if False] # total is reported below

active=list(requested); shed=[]
if not breaker_closed:
    cap=0.
def rerun():
    tt,vv,ii=simulate(active,cap,switching_event)
    return tt,vv,ii,analyze(tt,vv,ii)
t,v,i,m=rerun()
if shedding:
    for candidate in ["Resistive","Nonlinear","Inductive"]:
        if m["ir"]<=current_limit: break
        if candidate in active:
            active.remove(candidate); shed.append(candidate)
            t,v,i,m=rerun()

def detect_transients(time_axis, voltage_wave, current_wave):
    def residual(signal):
        window=max(5,int(fs/freq)//2*2+1)
        smooth=np.convolve(signal,np.ones(window)/window,mode="same")
        return signal-smooth

    voltage_residual=residual(voltage_wave)
    current_residual=residual(current_wave)
    combined=np.maximum(
        np.abs(voltage_residual)/(max(voltage,1.0)*.01),
        np.abs(current_residual)/.05)
    threshold=max(6*np.median(np.abs(combined)), 2.5)
    mask=combined>threshold
    edge=max(1,int(fs/freq))
    mask[:edge]=False
    mask[-edge:]=False
    indexes=np.flatnonzero(mask)
    if not len(indexes):
        return None
    groups=np.split(indexes,np.where(np.diff(indexes)>1)[0]+1)
    group=max(groups,key=lambda candidate: np.max(combined[candidate]))
    start,end=int(group[0]),int(group[-1])
    peak=int(group[np.argmax(combined[group])])
    padding=max(1,int(.002*fs))
    start=max(0,start-padding)
    end=min(len(time_axis)-1,end+padding)
    return {
        "start_index":start,
        "end_index":end,
        "peak_index":peak,
        "start_ms":float(time_axis[start]*1000),
        "end_ms":float(time_axis[end]*1000),
        "duration_ms":float((time_axis[end]-time_axis[start])*1000),
        "voltage_peak_V":float(np.max(np.abs(voltage_residual[start:end+1]))),
        "current_peak_A":float(np.max(np.abs(current_residual[start:end+1]))),
        "voltage_residual":voltage_residual,
        "current_residual":current_residual,
    }

transient=detect_transients(t,v,i)
if "transient_events" not in st.session_state:
    st.session_state.transient_events=[]
if transient and switching_event:
    st.session_state.transient_events.append({
        "Time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Operation":"Breaker/load/APFC switching",
        "Start (ms)":round(transient["start_ms"],3),
        "End (ms)":round(transient["end_ms"],3),
        "Duration (ms)":round(transient["duration_ms"],3),
        "Voltage peak (V)":round(transient["voltage_peak_V"],3),
        "Current peak (A)":round(transient["current_peak_A"],3),
    })

# Synthetic training data: demonstration only; replace with labelled real recordings for research claims.
@st.cache_resource
def train_model():
    rng=np.random.default_rng(7); X=[]; y=[]
    for cls in ["Resistive-like","Inductive-like","Capacitive-like","Nonlinear","Mixed/unknown"]:
        for _ in range(500):
            vr=rng.normal(230,5)
            if cls=="Resistive-like": ir=rng.uniform(.5,5); pf=rng.normal(.98,.015); ang=rng.normal(0,3); thd=rng.uniform(0,5)
            elif cls=="Inductive-like": ir=rng.uniform(.5,5); pf=rng.uniform(.55,.92); ang=rng.uniform(20,65); thd=rng.uniform(0,8)
            elif cls=="Capacitive-like": ir=rng.uniform(.3,4); pf=rng.uniform(.65,.98); ang=rng.uniform(-60,-10); thd=rng.uniform(0,8)
            elif cls=="Nonlinear": ir=rng.uniform(.3,5); pf=rng.uniform(.55,.98); ang=rng.uniform(-20,25); thd=rng.uniform(18,100)
            else: ir=rng.uniform(1,8); pf=rng.uniform(.45,.98); ang=rng.uniform(-45,55); thd=rng.uniform(5,45)
            X.append([vr,ir,vr*ir*pf,pf,ang,thd,thd*.5,thd*.25,thd*.12]); y.append(cls)
    model=make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=140,max_depth=12,random_state=7))
    model.fit(X,y); return model
model=train_model()

def signature_features(measurement):
    h1=max(float(measurement["ih"].iloc[0]["RMS magnitude"]),1e-9)
    return [measurement["vr"],measurement["ir"],measurement["p"],abs(measurement["pf"]),
            measurement["angle"],measurement["ithd"]]+[
        float(measurement["ih"].loc[measurement["ih"].Order==h,"RMS magnitude"].iloc[0]/h1*100)
        for h in [3,5,7]]

features=signature_features(m)
prediction=model.predict([features])[0]
confidence=float(np.max(model.predict_proba([features])[0])*100)

# NILM model: learn appliance presence from aggregate voltage/current signatures.
@st.cache_resource
def train_nilm_model(voltage_setting, freq_setting, fs_setting, window_setting):
    rng=np.random.default_rng(42)
    appliance_names=["Resistive","Inductive","Capacitive","Nonlinear"]
    feature_scale=np.array([.8,.04,8,.01,1.5,1.5,1.5,1.5,1.5])
    X=[]; y=[]
    for bits in product([0,1],repeat=4):
        loads=[name for name,on in zip(appliance_names,bits) if on]
        for capacitor_uF in [0,2,4,6,8,10,12,14]:
            training_t,training_v,training_i=simulate(loads,capacitor_uF)
            signature=signature_features(analyze(training_t,training_v,training_i))
            for _ in range(45):
                X.append(signature+rng.normal(0,feature_scale))
                y.append(bits)
    detector=MultiOutputClassifier(RandomForestClassifier(
        n_estimators=100,max_depth=14,random_state=42,n_jobs=-1))
    detector.fit(X,np.asarray(y))
    return detector

nilm_model=train_nilm_model(voltage,freq,fs,window_ms)
nilm_prediction=nilm_model.predict([features])[0]
nilm_confidence=np.array([
    np.max(estimator.predict_proba([features])[0])
    for estimator in nilm_model.estimators_
])*100

# ----------------------------- AI anomaly detection ---------------------------
# Synthetic normal-operation training data for demonstration only.
@st.cache_resource
def train_anomaly_detector():
    rng = np.random.default_rng(123)
    rows = []
    for _ in range(3000):
        vr = rng.normal(230, 4.0)
        ir = rng.uniform(0.15, 5.0)
        pf = rng.uniform(0.78, 1.0)
        angle = rng.uniform(-25, 35)
        thd = rng.uniform(0.2, 8.0)
        p = vr * ir * pf
        rows.append([vr, ir, p, pf, angle, thd,
                     thd*rng.uniform(.20,.55),
                     thd*rng.uniform(.08,.30),
                     thd*rng.uniform(.03,.18)])
    detector = IsolationForest(n_estimators=200, contamination=0.04, random_state=123)
    detector.fit(np.asarray(rows))
    return detector

anomaly_model = train_anomaly_detector()
anomaly_result = int(anomaly_model.predict([features])[0])  # 1=inlier, -1=outlier
anomaly_score = float(anomaly_model.decision_function([features])[0])

# Transparent threshold checks supplement the ML detector.
anomaly_reasons = []
if m["vr"] < 207 or m["vr"] > 253:
    anomaly_reasons.append("Voltage is outside the illustrative 207–253 V operating band.")
if m["ir"] > current_limit:
    anomaly_reasons.append(f"Current exceeds the configured limit ({current_limit:.2f} A).")
if m["ithd"] > thd_limit:
    anomaly_reasons.append(f"Current THD exceeds the configured limit ({thd_limit:.1f}%).")
if abs(m["angle"]) > 45:
    anomaly_reasons.append("Fundamental voltage-current phase displacement is unusually large.")
if anomaly_result == -1:
    anomaly_reasons.append("Isolation Forest classified the combined measurement features as unusual.")

is_anomaly = bool(anomaly_reasons)
anomaly_status = "ANOMALY DETECTED" if is_anomaly else "Normal-looking operating point"

if "energy_kwh" not in st.session_state: st.session_state.energy_kwh=0.0
if "events" not in st.session_state: st.session_state.events=[]
if "last_state" not in st.session_state: st.session_state.last_state=None
window_energy=max(m["p"],0)*(window_ms/1000)/3_600_000
st.session_state.energy_kwh+=window_energy
state=(tuple(active),cap,round(m["pf"],3),round(m["ithd"],2),tuple(shed))
if state!=st.session_state.last_state:
    st.session_state.events.append({"Time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Active loads":", ".join(active) or "None","Capacitor (µF)":cap,
        "Current (A)":round(m["ir"],3),"PF":round(m["pf"],3),
        "Current THD (%)":round(m["ithd"],2),"Shed":", ".join(shed) or "None"})
    st.session_state.last_state=state

alerts=[]
if not breaker_closed:
    alerts.append(("Main breaker open","Downstream loads are de-energized by operator command","warning"))
if active and m["vr"]<v_low: alerts.append(("Undervoltage",f"Voltage below {v_low:.1f} V","warning"))
if active and m["vr"]>v_high: alerts.append(("Overvoltage",f"Voltage above {v_high:.1f} V","error"))
if m["ir"]>current_limit: alerts.append(("Overcurrent",f"Current exceeds {current_limit:.1f} A","error"))
if m["ithd"]>thd_limit: alerts.append(("High current THD",f"Current THD exceeds {thd_limit:.1f}%","warning"))
if active and abs(m["pf"])<.8: alerts.append(("Poor PF","True PF magnitude is below 0.80","warning"))
if shed: alerts.append(("Load shedding","Disconnected: "+", ".join(shed),"warning"))
if is_anomaly: alerts.append(("AI anomaly detection","; ".join(anomaly_reasons),"warning"))
if apfc and cap: alerts.append(("APFC active",f"Connected total capacitor stage: {cap:g} µF","success"))
if apfc and m["q"]< -1: alerts.append(("Leading load","No capacitor added because the load is already leading","warning"))

st.subheader("Live measurements")
if active: st.info("Connected loads: "+", ".join(active))
else: st.warning("All loads are OFF. Supply voltage remains present; current and power are approximately zero.")
c=st.columns(5)
for col,label,val in zip(c,["Voltage RMS","Current RMS","True PF","Current THD","Voltage THD"],
                          [f'{m["vr"]:.2f} V',f'{m["ir"]:.3f} A',f'{abs(m["pf"]):.3f}',f'{m["ithd"]:.2f}%',f'{m["vthd"]:.2f}%']):
    col.metric(label,val)
c=st.columns(4)
for col,label,val in zip(c,["Active power","Apparent power","Fundamental Q","Displacement PF"],
                          [f'{m["p"]:.2f} W',f'{m["s"]:.2f} VA',f'{m["q"]:.2f} var',f'{abs(np.cos(np.deg2rad(m["angle"]))):.3f}']):
    col.metric(label,val)

tabs=st.tabs(["Waveforms & FFT","APFC & Load shedding","SCADA overview","Digital twin","ML identification","NILM","Transient detection","AI anomaly detection","Alerts & Events","Energy","Downloads & notes"])

with tabs[0]:
    end=min(len(t),int(3*fs/freq))
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,subplot_titles=("Voltage waveform","Current waveform"))
    fig.add_trace(go.Scatter(x=t[:end]*1000,y=v[:end],name="Voltage (V)"),row=1,col=1)
    fig.add_trace(go.Scatter(x=t[:end]*1000,y=i[:end],name="Current (A)"),row=2,col=1)
    fig.update_xaxes(title_text="Time (ms)",row=2,col=1)
    fig.update_layout(height=500,margin=dict(l=10,r=10,t=45,b=10))
    st.plotly_chart(fig,use_container_width=True)
    a,b=st.columns(2)
    for col,tab,title,unit in [(a,m["vh"],"Voltage FFT","V"),(b,m["ih"],"Current FFT","A")]:
        f=go.Figure(go.Bar(x=tab["Frequency (Hz)"],y=tab["RMS magnitude"]))
        f.update_layout(title=title,xaxis_title="Frequency (Hz)",yaxis_title=f"RMS ({unit})",height=350)
        col.plotly_chart(f,use_container_width=True)
    st.caption("THD = √(H₂² + … + H₄₀²) / H₁ × 100, calculated from FFT harmonic RMS magnitudes.")

with tabs[1]:
    st.subheader("Automatic power-factor correction")
    st.metric("APFC", "Enabled" if apfc else "Disabled")
    st.metric("Total selected capacitor stages",f"{cap:g} µF")
    st.write("Discrete simulated capacitor stages: 2, 4 and 8 µF. The controller selects the combination that most closely approaches the target displacement PF for lagging loads.")
    if apfc and not cap: st.info("No capacitor stage selected: load may be leading, near target, or available stages may not improve the target.")
    st.subheader("Automatic load shedding")
    st.write(f"Current threshold: **{current_limit:.2f} A**")
    if shed: st.warning("Automatically disconnected: "+", ".join(shed))
    elif shedding: st.success("No load needed to be shed.")
    else: st.info("Automatic shedding is disabled.")
    st.caption("Simulation priority: heater → nonlinear load → motor. Real systems require application-specific critical-load priorities.")

with tabs[2]:
    st.subheader("SCADA overview")
    st.caption("Supervisory view of the simulated feeder. Statuses are derived from the current breaker, load, APFC, shedding, and alarm state.")
    if not st.session_state.get("scada_acknowledged", False):
        st.session_state.scada_acknowledged = False
    scada_left, scada_right = st.columns([1.5, 1])
    with scada_left:
        nodes={"AC supply":(0,2),"Main breaker":(1.7,2),"Measurement":(3.4,2),"Load bus":(5.1,2),
               "Heater":(7.2,3.2),"Motor":(7.2,2.4),"Capacitive":(7.2,1.6),"Rectifier":(7.2,.8),
               "Capacitor bank":(5.1,.5)}
        edges=[("AC supply","Main breaker"),("Main breaker","Measurement"),
               ("Measurement","Load bus"),("Load bus","Heater"),("Load bus","Motor"),
               ("Load bus","Capacitive"),("Load bus","Rectifier"),("Measurement","Capacitor bank")]
        diagram=go.Figure()
        for source,target in edges:
            x0,y0=nodes[source]; x1,y1=nodes[target]
            diagram.add_trace(go.Scatter(x=[x0,x1],y=[y0,y1],mode="lines",
                line=dict(color="#22c55e" if breaker_closed and source != "Measurement" or source == "AC supply" else "#64748b",width=4),
                showlegend=False,hoverinfo="skip"))
        status_by_node={"Heater":"Resistive","Motor":"Inductive","Capacitive":"Capacitive","Rectifier":"Nonlinear"}
        for name,(x,y) in nodes.items():
            if name=="Main breaker":
                is_on=breaker_closed
            elif name=="Capacitor bank":
                is_on=cap>0
            elif name in status_by_node:
                is_on=status_by_node[name] in active
            else:
                is_on=True
            diagram.add_trace(go.Scatter(x=[x],y=[y],mode="markers+text",text=[name],
                textposition="bottom center",marker=dict(size=30,
                color="#22c55e" if is_on else "#64748b",line=dict(color="white",width=2)),
                hovertemplate=f"{name}: {'ON' if is_on else 'OFF'}<extra></extra>",showlegend=False))
        diagram.update_layout(height=430,xaxis=dict(visible=False,range=[-.5,8.5]),
            yaxis=dict(visible=False,range=[0,3.8]),plot_bgcolor="#111827",
            paper_bgcolor="#111827",font=dict(color="white"),margin=dict(l=5,r=5,t=10,b=35))
        st.plotly_chart(diagram,use_container_width=True)
        st.caption("Green = energized/active. Grey = open, switched off, shed, or unavailable.")
    with scada_right:
        st.markdown("#### Feeder status")
        st.metric("Main breaker", "CLOSED" if breaker_closed else "OPEN")
        st.metric("Load bus", "ENERGIZED" if breaker_closed else "DE-ENERGIZED")
        st.metric("Capacitor bank", f"{cap:g} µF" if cap else "DISCONNECTED")
        if alerts:
            st.markdown("#### Active alarms")
            alarm_rows=[{"Alarm":title,"Severity":level.upper(),"State":
                         "ACKNOWLEDGED" if st.session_state.scada_acknowledged else "ACTIVE"}
                        for title,_,level in alerts]
            st.dataframe(pd.DataFrame(alarm_rows),use_container_width=True,hide_index=True)
            if st.button("Acknowledge active alarms",key="scada_acknowledge"):
                st.session_state.scada_acknowledged=True
                st.rerun()
        else:
            st.success("No active alarms.")
            st.session_state.scada_acknowledged=False
    st.markdown("#### Equipment status")
    equipment_rows=[
        {"Equipment":"Main breaker","Command":"Close" if breaker_closed else "Open",
         "Status":"CLOSED" if breaker_closed else "OPEN","Condition":"Operator controlled"},
        {"Equipment":"Heater","Command":"ON" if sw_r else "OFF",
         "Status":"RUNNING" if "Resistive" in active else ("SHED" if "Resistive" in shed else "STOPPED"),
         "Condition":"Manual load switch"},
        {"Equipment":"Motor","Command":"ON" if sw_l else "OFF",
         "Status":"RUNNING" if "Inductive" in active else ("SHED" if "Inductive" in shed else "STOPPED"),
         "Condition":"Manual load switch"},
        {"Equipment":"Capacitive load","Command":"ON" if sw_c else "OFF",
         "Status":"RUNNING" if "Capacitive" in active else ("SHED" if "Capacitive" in shed else "STOPPED"),
         "Condition":"Manual load switch"},
        {"Equipment":"Rectifier","Command":"ON" if sw_n else "OFF",
         "Status":"RUNNING" if "Nonlinear" in active else ("SHED" if "Nonlinear" in shed else "STOPPED"),
         "Condition":"Manual load switch"},
        {"Equipment":"APFC capacitor bank","Command":"AUTO" if apfc else "OFF",
         "Status":f"CONNECTED ({cap:g} µF)" if cap else "DISCONNECTED",
         "Condition":"Automatic PF correction"},
    ]
    st.dataframe(pd.DataFrame(equipment_rows),use_container_width=True,hide_index=True)

with tabs[3]:
    st.subheader("Single-line digital twin")
    nodes={"AC supply":(0,2),"Voltage sensor":(1.5,2.8),"Current sensor":(1.5,1.2),"Controller":(3.2,2),
           "Load bus":(5,2),"Resistive":(7,3.2),"Inductive":(7,2.4),"Capacitive":(7,1.6),"Nonlinear":(7,.8),"APFC bank":(5,.5)}
    edges=[("AC supply","Voltage sensor"),("AC supply","Current sensor"),("Voltage sensor","Controller"),
           ("Current sensor","Controller"),("Controller","Load bus"),("Load bus","Resistive"),
           ("Load bus","Inductive"),("Load bus","Capacitive"),("Load bus","Nonlinear"),("Controller","APFC bank")]
    f=go.Figure()
    for a,b in edges:
        x0,y0=nodes[a];x1,y1=nodes[b]
        f.add_trace(go.Scatter(x=[x0,x1],y=[y0,y1],mode="lines",line=dict(color="#94a3b8",width=2),showlegend=False,hoverinfo="skip"))
    for name,(x,y) in nodes.items():
        on=(name=="AC supply" or name=="Voltage sensor" or name=="Current sensor" or name=="Controller" or name=="Load bus")
        on=on or (name in active) or (name=="APFC bank" and cap>0)
        f.add_trace(go.Scatter(x=[x],y=[y],mode="markers+text",text=[name],textposition="bottom center",
            marker=dict(size=28,color="#16a34a" if on else "#94a3b8",line=dict(color="white",width=2)),
            hovertemplate=f"{name}: {'ON' if on else 'OFF'}<extra></extra>",showlegend=False))
    f.update_layout(height=450,xaxis=dict(visible=False,range=[-.5,8]),yaxis=dict(visible=False,range=[0,3.8]),plot_bgcolor="white",margin=dict(l=5,r=5,t=10,b=20))
    st.plotly_chart(f,use_container_width=True)
    st.caption("Green = active/available; grey = switched off. This is a schematic visualization, not a wiring diagram.")

with tabs[4]:
    st.subheader("Random Forest load characterization")
    st.metric("Predicted load behaviour",prediction,f"synthetic-model score: {confidence:.1f}%")
    st.write("Features: Vrms, Irms, active power, true PF, fundamental phase angle, current THD and H3/H5/H7 ratios.")
    scores=pd.DataFrame({"Class":model.classes_,"Model score (%)":model.predict_proba([features])[0]*100}).sort_values("Model score (%)")
    f=go.Figure(go.Bar(x=scores["Model score (%)"],y=scores["Class"],orientation="h"))
    f.update_layout(height=330,xaxis_title="Model score (%)")
    st.plotly_chart(f,use_container_width=True)
    st.warning("This model is trained on synthetic feature data for demonstrating the workflow. Its score is not real-world accuracy. Replace the training data with labelled, calibrated hardware recordings and test on separate recording sessions.")

with tabs[5]:
    st.subheader("Non-Intrusive Load Monitoring")
    st.write("The NILM model infers which appliances are operating from the aggregate voltage/current signature at the common measurement point.")
    nilm_names=["Heater","Motor","Capacitive load","Rectifier"]
    actual_nilm={"Heater":"Resistive" in active,"Motor":"Inductive" in active,
                 "Capacitive load":"Capacitive" in active,"Rectifier":"Nonlinear" in active}
    nilm_rows=[]
    for name,model_name,detected,score in zip(nilm_names,
            ["Resistive","Inductive","Capacitive","Nonlinear"],
            nilm_prediction,nilm_confidence):
        nilm_rows.append({"Appliance":name,"Inferred state":"ON" if detected else "OFF",
                          "Confidence (%)":round(float(score),1),
                          "Simulated state":"ON" if actual_nilm[name] else "OFF",
                          "Match":"Yes" if bool(detected)==actual_nilm[name] else "No"})
    st.dataframe(pd.DataFrame(nilm_rows),use_container_width=True,hide_index=True)
    detected_names=[name for name,detected in zip(nilm_names,nilm_prediction) if detected]
    a,b,c=st.columns(3)
    a.metric("Appliances inferred",str(len(detected_names)))
    b.metric("Aggregate current",f'{m["ir"]:.3f} A')
    c.metric("Aggregate active power",f'{m["p"]:.2f} W')
    if detected_names:
        st.info("Inferred operating loads: "+", ".join(detected_names))
    else:
        st.info("No appliance activity inferred from the aggregate signature.")
    st.caption("Features: Vrms, Irms, active power, PF, phase angle, current THD, and H3/H5/H7 ratios. Training data is synthetic and includes simulated APFC capacitor stages.")
    st.warning("Research limitation: this NILM result demonstrates aggregate-signature classification, not validated appliance disaggregation. Replace synthetic training data with labelled common-point recordings and evaluate on separate households or operating sessions.")

with tabs[6]:
    st.subheader("Transient detection")
    st.write("The detector looks for short-duration deviations from the local voltage and current waveform baseline. Switching the breaker, a load, or APFC creates a simulated damped transient for demonstration.")
    if transient:
        st.error("Transient detected in the current measurement window.")
        a,b,c=st.columns(3)
        a.metric("Duration",f'{transient["duration_ms"]:.3f} ms')
        b.metric("Voltage peak",f'{transient["voltage_peak_V"]:.3f} V')
        c.metric("Current peak",f'{transient["current_peak_A"]:.3f} A')
        start=max(0,transient["start_index"]-int(.004*fs))
        end=min(len(t),transient["end_index"]+int(.004*fs)+1)
        transient_fig=make_subplots(rows=2,cols=1,shared_xaxes=True,
                                    subplot_titles=("Transient voltage","Transient current"))
        transient_fig.add_trace(go.Scatter(x=t[start:end]*1000,y=v[start:end],
                                            name="Voltage (V)"),row=1,col=1)
        transient_fig.add_trace(go.Scatter(x=t[start:end]*1000,y=i[start:end],
                                            name="Current (A)"),row=2,col=1)
        transient_fig.update_xaxes(title_text="Time (ms)",row=2,col=1)
        transient_fig.update_layout(height=500,margin=dict(l=10,r=10,t=45,b=10))
        st.plotly_chart(transient_fig,use_container_width=True)
    else:
        st.success("No transient detected in the current measurement window.")
        st.info("Change the main breaker, a manual load switch, or APFC setting to create a switching event.")
    st.markdown("#### Transient event log")
    if st.session_state.transient_events:
        st.dataframe(pd.DataFrame(st.session_state.transient_events[::-1]),
                     use_container_width=True,hide_index=True)
    else:
        st.caption("No transient events recorded yet.")
    st.caption("Detection is based on residual voltage/current peaks after local baseline removal. This is a simulation aid, not a protection relay.")

with tabs[7]:
    st.subheader("AI-based electrical anomaly detection")
    st.write("Isolation Forest checks whether the current electrical feature vector looks unusual compared with its synthetic normal-operation training distribution.")
    if is_anomaly:
        st.error("⚠️ " + anomaly_status)
    else:
        st.success("✓ " + anomaly_status)
    a, b = st.columns(2)
    a.metric("Isolation Forest decision score", f"{anomaly_score:.4f}")
    b.metric("AI classification", "Anomaly" if anomaly_result == -1 else "Inlier")
    st.caption("Lower decision scores indicate more unusual observations; a negative score is generally classified as an outlier.")
    anomaly_feature_table = pd.DataFrame({
        "Feature": ["Voltage RMS", "Current RMS", "Active power", "True PF magnitude",
                    "Fundamental phase angle", "Current THD", "H3/H1", "H5/H1", "H7/H1"],
        "Value": features,
        "Unit": ["V", "A", "W", "ratio", "degrees", "%", "%", "%", "%"]
    })
    st.markdown("#### Features evaluated")
    st.dataframe(anomaly_feature_table, use_container_width=True, hide_index=True)
    st.markdown("#### Reasons / indicators")
    if anomaly_reasons:
        for reason in anomaly_reasons:
            st.warning(reason)
    else:
        st.info("No configured threshold violation or Isolation Forest outlier was detected.")
    st.markdown("#### How to test")
    st.write("Switch ON multiple loads, enable the nonlinear load, adjust supply voltage and change the current threshold. Observe the feature values and anomaly result.")
    st.warning("Research limitation: this detector is trained on synthetic normal-operation data. It demonstrates the workflow but is not validated for real installations. Train using calibrated healthy measurements and validate against separately recorded abnormal cases. An anomaly flag does not diagnose a specific fault.")

with tabs[8]:
    st.subheader("Active alerts")
    if not alerts: st.success("No configured alerts active.")
    for title,msg,level in alerts:
        if level=="error": st.error(f"**{title}:** {msg}")
        elif level=="warning": st.warning(f"**{title}:** {msg}")
        else: st.success(f"**{title}:** {msg}")
    st.subheader("Event history")
    if st.session_state.events: st.dataframe(pd.DataFrame(st.session_state.events[::-1]),use_container_width=True,hide_index=True)
    else: st.caption("No events recorded yet.")

with tabs[9]:
    st.subheader("Energy analytics")
    cost=st.session_state.energy_kwh*tariff
    st.metric("Cumulative simulated energy",f'{st.session_state.energy_kwh:.9f} kWh')
    st.metric("Estimated cost",f"₹{cost:.6f}")
    st.metric("Energy in this measurement window",f"{max(m['p'],0)*(window_ms/1000)/3_600_000:.10f} kWh")
    st.caption("Each Streamlit rerun adds one measurement window. This is a demonstration counter, not a real-time clock-based energy meter.")
    if st.button("Reset energy counter"):
        st.session_state.energy_kwh=0.0
        st.rerun()

with tabs[10]:
    st.subheader("Measurement method")
    st.write(f"- Sampling: {fs:,} samples/s/channel; window: {window_ms} ms; samples: {len(t):,}.")
    st.write(f"- Nominal FFT-bin spacing: {fs/len(t):.2f} Hz; Nyquist frequency: {fs/2:.0f} Hz.")
    st.write("- THD range: 2nd through 40th harmonic.")
    st.write("- True PF = average(v × i) / (Vrms × Irms); displacement PF is calculated from fundamental phase.")
    st.write("- Fundamental reactive power is estimated from fundamental voltage/current and phase.")
    wave=pd.DataFrame({"time_s":t,"voltage_V":v,"current_A":i})
    summary=pd.DataFrame([{"loads":", ".join(active) or "None","shed":", ".join(shed) or "None","capacitor_uF":cap,
        "Vrms_V":m["vr"],"Irms_A":m["ir"],"P_W":m["p"],"S_VA":m["s"],"true_PF":m["pf"],
        "phase_deg":m["angle"],"fundamental_Q_var":m["q"],"VTHD_percent":m["vthd"],"ITHD_percent":m["ithd"],
        "ML_prediction":prediction,"ML_score_percent":confidence}])
    st.download_button("Download waveform CSV",wave.to_csv(index=False).encode(),"power_quality_waveforms.csv","text/csv")
    st.download_button("Download summary CSV",summary.to_csv(index=False).encode(),"power_quality_summary.csv","text/csv")
    if st.session_state.events:
        st.download_button("Download event log CSV",pd.DataFrame(st.session_state.events).to_csv(index=False).encode(),"power_quality_events.csv","text/csv")
    if st.session_state.transient_events:
        st.download_button("Download transient log CSV",pd.DataFrame(st.session_state.transient_events).to_csv(index=False).encode(),"power_quality_transients.csv","text/csv")
    st.warning("Software simulation only. Never connect a PC, ESP32, ADC or breadboard directly to mains. Physical work requires isolated, correctly rated sensing, protection, enclosure and qualified supervision. Capacitor banks require capacitor-duty switching and discharge provisions.")

st.divider()
st.caption("Educational simulation prototype. Validate calculations against calibrated instruments before engineering use.")
