# Intelligent Single-Phase Power Quality Monitoring Simulator

An interactive [Streamlit](https://streamlit.io/) simulation for exploring single-phase power-quality monitoring, automatic power-factor correction (APFC), load shedding, alerts, energy tracking, and synthetic load characterization.

The simulator models a configurable AC supply and four illustrative load types:

- Resistive heater
- Inductive motor
- Capacitive load
- Nonlinear rectifier load

It calculates voltage and current RMS values, active and apparent power, true and displacement power factor, fundamental reactive power, FFT harmonic magnitudes, and voltage/current THD.

> **Safety:** This is software only. Never connect a PC, ESP32, ADC, breadboard, or other unevaluated circuit directly to mains. Any physical prototype requires isolated, correctly rated sensing, protection, enclosure, calibration, and qualified supervision. Begin with an isolated low-voltage AC source.

## Features

- Interactive supply voltage, frequency, sampling-rate, and measurement-window controls
- Manual ON/OFF switching for resistive, inductive, capacitive, and nonlinear loads
- Voltage/current waveform plots and harmonic FFT bar charts
- True PF and fundamental displacement PF measurements
- Discrete APFC using simulated 2, 4, and 8 µF capacitor stages
- Automatic load shedding with configurable current limit
- Configurable undervoltage, overvoltage, overcurrent, current-THD, and poor-PF alerts
- Digital-twin-style single-line visualization of the simulated system
- Random Forest load-behaviour identification using synthetic training data
- Session energy and estimated tariff tracking
- CSV downloads for waveforms, measurement summaries, and event history

## Requirements

- Python 3.10 or later
- A modern web browser
- The packages listed in [`requirements.txt`](requirements.txt)

## Run locally

From the project directory:

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit normally opens the application at <http://localhost:8501>. If it does not open automatically, copy that address into a browser.

To stop the application, press `Ctrl+C` in the terminal.

## Using the dashboard

All controls are in the left sidebar.

### Supply and acquisition

- **Supply voltage:** 180–250 V RMS
- **Frequency:** 45–55 Hz from the available preset values
- **Sampling rate:** 4, 8, 12, or 16 kS/s per channel
- **Measurement window:** 100, 200, or 400 ms

### Manual loads

Enable any combination of the four load switches:

- **Resistive:** approximately in-phase current
- **Inductive:** lagging fundamental current
- **Capacitive:** leading fundamental current
- **Nonlinear:** rectifier-like current with 3rd, 5th, and 7th harmonics

The simulation recalculates whenever a control changes. With every load disabled, the supply voltage remains present while current and power are approximately zero.

### Automatic control

- **Automatic PF correction:** selects a combination of simulated 2, 4, and 8 µF capacitor stages to approach the target displacement PF for lagging loads.
- **Automatic load shedding:** disconnects loads when current exceeds the configured limit. The demonstration priority is heater, nonlinear load, then motor.

### Alerts and energy

Set the tariff and alert thresholds in the sidebar. Each Streamlit rerun adds one simulated measurement window to the session energy counter. Use **Reset energy counter** in the **Energy** tab to clear it.

## Dashboard tabs

- **Waveforms & FFT:** Displays up to three cycles of voltage/current waveforms and harmonic magnitudes through the 40th harmonic.
- **APFC & Load shedding:** Shows selected capacitor stages, PF-correction status, current threshold, and shed loads.
- **Digital twin:** Shows the simulated supply, sensors, controller, load bus, loads, and APFC bank.
- **ML identification:** Displays the predicted load behaviour and model scores.
- **Alerts & Events:** Shows active threshold alerts and the session event history.
- **Energy:** Shows cumulative simulated energy, estimated cost, and the current-window energy.
- **Downloads & notes:** Provides measurement definitions and CSV downloads.

## Measurement and simulation details

- Samples are generated as `n = sampling_rate × window_ms / 1000`.
- FFT bin spacing is `sampling_rate / number_of_samples`; at the default 8 kS/s and 200 ms settings it is 5 Hz.
- Harmonic magnitudes are calculated from a Hann-windowed FFT.
- THD is calculated from the 2nd through 40th harmonic relative to the fundamental:

  `THD = sqrt(H2² + H3² + ... + H40²) / H1 × 100`

- True PF is calculated as `average(voltage × current) / (Vrms × Irms)`.
- Displacement PF is calculated from the phase difference between the fundamental voltage and current.
- Fundamental reactive power is estimated from the fundamental RMS values and phase angle.
- Downloaded waveform data contains `time_s`, `voltage_V`, and `current_A` columns.
- Downloaded summary data includes electrical measurements, THD, APFC/load-shedding state, and ML output.

## ML demonstration limitations

The load-characterization model is a demonstration Random Forest trained at runtime on generated feature ranges. Its score is a synthetic model confidence, not measured accuracy or a claim about real equipment.

For research or deployment, replace the generated training data with labelled, calibrated recordings collected under representative operating conditions, and evaluate against separate recording sessions.

## Project files

```text
Power_Quality_Monitoring_Simulator/
├── app.py
├── requirements.txt
└── README.md
```

## Troubleshooting

### `ModuleNotFoundError`

Make sure the virtual environment is active and reinstall the dependencies:

```powershell
python -m pip install -r requirements.txt
```

### PowerShell does not allow script activation

Instead of activating the environment, invoke its Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### The browser page is not available

Check the terminal for the Streamlit URL and open it manually. If port 8501 is already in use, start Streamlit on another port:

```powershell
streamlit run app.py --server.port 8502
```
