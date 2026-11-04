import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
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

def simulate(loads, cap_uF=0):
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

requested=[name for name,on in [("Resistive",sw_r),("Inductive",sw_l),("Capacitive",sw_c),("Nonlinear",sw_n)] if on]
t,v,i=simulate(requested)
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
def rerun():
    tt,vv,ii=simulate(active,cap)
    return tt,vv,ii,analyze(tt,vv,ii)
t,v,i,m=rerun()
if shedding:
    for candidate in ["Resistive","Nonlinear","Inductive"]:
        if m["ir"]<=current_limit: break
        if candidate in active:
            active.remove(candidate); shed.append(candidate)
            t,v,i,m=rerun()

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
h1=max(float(m["ih"].iloc[0]["RMS magnitude"]),1e-9)
features=[m["vr"],m["ir"],m["p"],abs(m["pf"]),m["angle"],m["ithd"]]+[
    float(m["ih"].loc[m["ih"].Order==h,"RMS magnitude"].iloc[0]/h1*100) for h in [3,5,7]]
prediction=model.predict([features])[0]
confidence=float(np.max(model.predict_proba([features])[0])*100)

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
if active and m["vr"]<v_low: alerts.append(("Undervoltage",f"Voltage below {v_low:.1f} V","warning"))
if active and m["vr"]>v_high: alerts.append(("Overvoltage",f"Voltage above {v_high:.1f} V","error"))
if m["ir"]>current_limit: alerts.append(("Overcurrent",f"Current exceeds {current_limit:.1f} A","error"))
if m["ithd"]>thd_limit: alerts.append(("High current THD",f"Current THD exceeds {thd_limit:.1f}%","warning"))
if active and abs(m["pf"])<.8: alerts.append(("Poor PF","True PF magnitude is below 0.80","warning"))
if shed: alerts.append(("Load shedding","Disconnected: "+", ".join(shed),"warning"))
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

tabs=st.tabs(["Waveforms & FFT","APFC & Load shedding","Digital twin","ML identification","Alerts & Events","Energy","Downloads & notes"])

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

with tabs[3]:
    st.subheader("Random Forest load characterization")
    st.metric("Predicted load behaviour",prediction,f"synthetic-model score: {confidence:.1f}%")
    st.write("Features: Vrms, Irms, active power, true PF, fundamental phase angle, current THD and H3/H5/H7 ratios.")
    scores=pd.DataFrame({"Class":model.classes_,"Model score (%)":model.predict_proba([features])[0]*100}).sort_values("Model score (%)")
    f=go.Figure(go.Bar(x=scores["Model score (%)"],y=scores["Class"],orientation="h"))
    f.update_layout(height=330,xaxis_title="Model score (%)")
    st.plotly_chart(f,use_container_width=True)
    st.warning("This model is trained on synthetic feature data for demonstrating the workflow. Its score is not real-world accuracy. Replace the training data with labelled, calibrated hardware recordings and test on separate recording sessions.")

with tabs[4]:
    st.subheader("Active alerts")
    if not alerts: st.success("No configured alerts active.")
    for title,msg,level in alerts:
        if level=="error": st.error(f"**{title}:** {msg}")
        elif level=="warning": st.warning(f"**{title}:** {msg}")
        else: st.success(f"**{title}:** {msg}")
    st.subheader("Event history")
    if st.session_state.events: st.dataframe(pd.DataFrame(st.session_state.events[::-1]),use_container_width=True,hide_index=True)
    else: st.caption("No events recorded yet.")

with tabs[5]:
    st.subheader("Energy analytics")
    cost=st.session_state.energy_kwh*tariff
    st.metric("Cumulative simulated energy",f'{st.session_state.energy_kwh:.9f} kWh')
    st.metric("Estimated cost",f"₹{cost:.6f}")
    st.metric("Energy in this measurement window",f"{max(m['p'],0)*(window_ms/1000)/3_600_000:.10f} kWh")
    st.caption("Each Streamlit rerun adds one measurement window. This is a demonstration counter, not a real-time clock-based energy meter.")
    if st.button("Reset energy counter"):
        st.session_state.energy_kwh=0.0
        st.rerun()

with tabs[6]:
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
    st.warning("Software simulation only. Never connect a PC, ESP32, ADC or breadboard directly to mains. Physical work requires isolated, correctly rated sensing, protection, enclosure and qualified supervision. Capacitor banks require capacitor-duty switching and discharge provisions.")

st.divider()
st.caption("Educational simulation prototype. Validate calculations against calibrated instruments before engineering use.")
