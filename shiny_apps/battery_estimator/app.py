import numpy as np
import matplotlib.pyplot as plt
from shiny.express import ui, input, render

## Battery Models
def ocv_linear(soc):
    return 2.6 + 1.1 * soc # playing with linear approximation

def ocv_nonlinear(soc):
    return 2.6 + 2.35 * soc - 3.75 * soc**2 + 2.5 * soc**3


## UI
ui.page_opts(title="Battery SOC Estimator", fillable=True)

with ui.sidebar():
    ui.h4("Simulation Controls")

    ui.input_selectize(
        "mode", 
        "Estimator Type",
        [
            "Fixed-Gain Observer",
            "KF",
        ],
    )
    
    ui.input_selectize(
        "model", 
        "Prediction Model",
        [
            "True System",
            "Linear Approximation",
        ],
    )

    ui.input_slider("I", "Battery Current (A)", -2.2, 2.2, -2.2)
    ui.input_slider("SOC_0", "Initial SOC", 0.0, 1.0, 0.9)
    ui.input_slider("duration", "Time Duration (m)", 1, 65, 60)

    # Fixed-gain observer only
    with ui.panel_conditional("input.mode !== 'KF'"):
        ui.input_slider("K", "Observer Gain K", 0.0, 0.1, 0.01)

    with ui.panel_conditional("input.mode === 'KF'"):
        ui.input_slider(
            "logQk",
            "Process Noise \n(log10 Q)",
            min=-6,
            max=-3,
            value=-6,
            step=0.1
        )

        ui.input_slider(
            "logRk",
            "Measurement Noise \n(log10 R)",
            min=-4,
            max=-1,
            value=-2,
            step=0.1
        )



with ui.card(full_screen=True):
    ui.h3("State of Charge Estimation")

    @render.plot
    def soc_plot():
        # Parameters
        Q = 7920
        Rs = 0.05
        dt = 0.1
        t = np.arange(0, 3600, dt)
        
        #setup arrays
        v_true = np.zeros_like(t)
        v_meas = np.zeros_like(t)
        soc_true = np.zeros_like(t)
        soc_hat = np.zeros_like(t)
        
        #global inputs
        mode = input.mode()
        model = input.model()
        
        I = input.I() #current
        soc_true[0] = input.SOC_0()
        
        Vp = 0.0                 # polarization voltage
        tau = 40.0               # RC time constant (s)
        voltage_noise = 0.1           # voltage noise  (V)
        current_noise = 0.01            # current noise  (A)


        soc_hat[0] = 0.5  # initial estimate
        # EKF variables
        P = 0.01
    

        for k in range(1, len(t)):
            # True System
            soc_true[k] = soc_true[k - 1] + (I / Q) * dt
            soc_true[k] = np.clip(soc_true[k], 0.0, 1.0) 


            # True Voltage
            v_true[k] = ocv_nonlinear(soc_true[k]) + Rs * I
            v_meas[k] = v_true[k] + np.random.normal(0, voltage_noise)

            # Estimator
            soc_pred = soc_hat[k - 1] + ((I + np.random.normal(0, current_noise))/ Q) * dt

            if mode == "KF":
                Qk = 10 ** input.logQk() # process noise covariance
                Rk = 10 ** input.logRk() # measurement noise covariance


                # Prediction
                # v = 2.6 + H*soc + Rs*I
                H = 1.1
                P = P + Qk
                k_gain = P * H / (H * P * H + Rk)
                
                if model == "True System":
                    H = 2.35 - 7.5 * soc_pred + 7.5 * soc_pred**2
                    v_est = ocv_nonlinear(soc_pred) + Rs * I
                else:
                    v_est = ocv_linear(soc_pred) + Rs * I
                
                soc_hat[k] = soc_pred + k_gain * (v_meas[k] - v_est)
                soc_hat[k] = np.clip(soc_hat[k], 0.0, 1.0)

                # covariance update
                P = (1 - k_gain * H) * P

            else:
                K = input.K()
                
                if model == "True System":
                    v_est = ocv_nonlinear(soc_hat[k - 1]) + Rs * I
                else:
                    v_est = ocv_linear(soc_hat[k - 1]) + Rs * I
                soc_hat[k] = soc_pred + K * (v_meas[k] - v_est)
                soc_hat[k] = np.clip(soc_hat[k], 0.0, 1.0)


        ## Plot
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(8, 10), dpi=100
        )
        
        time_index = input.duration() * 60 *10

        t_min = t / 60  # seconds → minutes

        # SOC plot
        ax1.plot(
            t_min[1:time_index],
            soc_true[1:time_index],
            label="True SOC",
            linewidth=2
        )
        ax1.plot(
            t_min[1:time_index],
            soc_hat[1:time_index],
            label="Estimated SOC",
            linewidth=2
        )

        ax1.set_ylim(0, 1)
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("SOC")
        ax1.set_title(mode)
        ax1.grid(True)
        ax1.legend()


        # Voltage vs True SOC
        ax2.scatter(
                soc_true[1:time_index],
                v_meas[1:time_index],
                s=8,
                alpha=0.5,
                label="Measured Voltage"
            )
        ax2.plot(soc_true[1:time_index], v_true[1:time_index], label="True Voltage", linewidth=2, color="orange")

        ax2.set_xlabel("True SOC")
        ax2.set_ylabel("Voltage (V)")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(2, 4)
        ax2.grid(True)
        ax2.legend()

