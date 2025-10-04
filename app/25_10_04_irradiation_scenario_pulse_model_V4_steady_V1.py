#Author: Julijan Peric
#Project: KATANA - water activation model - pulse operation
#Version: 1 (KATANA pulse paper) -> from V7
# gif iradiation scenario maker is added
#Date: 04.10.2025
#pulse scenario: V4 25.10.2024
#steady state scenario: V1 25.10.2024
#ToDo:
# -steady state irradiation scenario - real reactor power (not that important)
# 
# -flow rate scenario
#   For now only one pump start and stop is possible
#   -multiple pump start and stop
#   -flow rate real data (not that easy :/)
##################################################################
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import os
import matplotlib.colors as colors





# funtions
#irradiation in irradiation snail
def irradiation(matrix_irradiation_snail,N16_RR,delta_time,lambdaN16,irradiation_power):
    irradiation_power = irradiation_power*0.25 #MW to 250kW
    #print(irradiation_power)
    for j in range(6):
        for i in range(22):
            matrix_irradiation_snail[j][i]=(matrix_irradiation_snail[j][i]*math.exp(-lambdaN16*delta_time)+(irradiation_power*N16_RR[i]*(1-math.exp(-lambdaN16*delta_time))))
    return matrix_irradiation_snail
#transport of the water
def transport(matrix_irradiation_snail,matrix_measurement_snail,lambdaN16,delta_time,n_voxels_pipe1,n_voxels_pipe2,matrix_transport_pipe1,matrix_transport_pipe2):
    matrix_irradiation_snailt1 = np.zeros((6,22))
    matrix_measurement_snailt1 = np.zeros((6,22))
    matrix_transport_pipe1t1 = np.zeros((1,n_voxels_pipe1))
    matrix_transport_pipe2t1 = np.zeros((1,n_voxels_pipe2))
    ### irradiation snail ###
    for i in range(22):
        matrix_irradiation_snailt1[0][i-1]=matrix_irradiation_snail[0][i]
        matrix_irradiation_snailt1[2][i-1]=matrix_irradiation_snail[2][i]
        matrix_irradiation_snailt1[4][i-1]=matrix_irradiation_snail[4][i]
    for i in range(21):
        matrix_irradiation_snailt1[1][i+1]=matrix_irradiation_snail[1][i]
        matrix_irradiation_snailt1[3][i+1]=matrix_irradiation_snail[3][i]
        matrix_irradiation_snailt1[5][i+1]=matrix_irradiation_snail[5][i]
    #
    matrix_irradiation_snailt1[1][0]=matrix_irradiation_snail[0][0]
    matrix_irradiation_snailt1[2][21]=matrix_irradiation_snail[1][21]
    matrix_irradiation_snailt1[3][0]=matrix_irradiation_snail[2][0]
    matrix_irradiation_snailt1[4][21]=matrix_irradiation_snail[3][21]
    matrix_irradiation_snailt1[5][0]=matrix_irradiation_snail[4][0]

    ### measurement snail ###
    for i in range(21):
        matrix_measurement_snailt1[0][i+1]=matrix_measurement_snail[0][i]*math.exp(-lambdaN16*delta_time)
        matrix_measurement_snailt1[2][i+1]=matrix_measurement_snail[2][i]*math.exp(-lambdaN16*delta_time)
        matrix_measurement_snailt1[4][i+1]=matrix_measurement_snail[4][i]*math.exp(-lambdaN16*delta_time)
    for i in range(22):
        matrix_measurement_snailt1[1][i-1]=matrix_measurement_snail[1][i]*math.exp(-lambdaN16*delta_time)
        matrix_measurement_snailt1[3][i-1]=matrix_measurement_snail[3][i]*math.exp(-lambdaN16*delta_time)
        matrix_measurement_snailt1[5][i-1]=matrix_measurement_snail[5][i]*math.exp(-lambdaN16*delta_time)
    #
    matrix_measurement_snailt1[1][21]=matrix_measurement_snail[0][21]*math.exp(-lambdaN16*delta_time)
    matrix_measurement_snailt1[2][0]=matrix_measurement_snail[1][0]*math.exp(-lambdaN16*delta_time)
    matrix_measurement_snailt1[3][21]=matrix_measurement_snail[2][21]*math.exp(-lambdaN16*delta_time)
    matrix_measurement_snailt1[4][0]=matrix_measurement_snail[3][0]*math.exp(-lambdaN16*delta_time)
    matrix_measurement_snailt1[5][21]=matrix_measurement_snail[4][21]*math.exp(-lambdaN16*delta_time)

    ### transfere pipes ###
    
    for i in range(n_voxels_pipe1-1):
        matrix_transport_pipe1t1[0][i+1]=matrix_transport_pipe1[0][i]*math.exp(-lambdaN16*delta_time)
    for i in range(n_voxels_pipe2):
        matrix_transport_pipe2t1[0][i-1]=matrix_transport_pipe2[0][i]*math.exp(-lambdaN16*delta_time)

    matrix_measurement_snailt1[0][0]=matrix_transport_pipe1[0][n_voxels_pipe1-1]*math.exp(-lambdaN16*delta_time) #from transport pipe1 last voxel to measurement volume first voxel
    matrix_transport_pipe1t1[0][0]=matrix_irradiation_snail[5][21]*math.exp(-lambdaN16*delta_time) #from irradiation volume first voxel of transport pipe 1
    matrix_irradiation_snailt1[0][21]=matrix_transport_pipe2[0][0]*math.exp(-lambdaN16*delta_time) #from transport pipe 2 lastv voxel to irradiation volume first voxel
    matrix_transport_pipe2t1[0][n_voxels_pipe2-1]=matrix_measurement_snail[5][0]*math.exp(-lambdaN16*delta_time) #from measurement volume last voxel to transport pipe 2 first voxel

    return matrix_irradiation_snailt1,matrix_measurement_snailt1,matrix_transport_pipe1t1,matrix_transport_pipe2t1
#decay of water when pump stops
def decay(matrix_irradiation_snail,matrix_measurement_snail,lambdaN16,delta_time,n_voxels_pipe1,n_voxels_pipe2,matrix_transport_pipe1,matrix_transport_pipe2):
    ### irradiation snail ###
    for i in range(22):
        for j in range(6):
            #matrix_irradiation_snail[j][i] = matrix_irradiation_snail[j][i]*math.exp(-lambdaN16*delta_time)
            matrix_measurement_snail[j][i] = matrix_measurement_snail[j][i]*math.exp(-lambdaN16*delta_time)

    ### transfere pipes ###
    
    for i in range(n_voxels_pipe1):
        matrix_transport_pipe1[0][i] = matrix_transport_pipe1[0][i]*math.exp(-lambdaN16*delta_time)
    for i in range(n_voxels_pipe2):
        matrix_transport_pipe2[0][i] = matrix_transport_pipe2[0][i]*math.exp(-lambdaN16*delta_time)

    

    return matrix_irradiation_snail,matrix_measurement_snail,matrix_transport_pipe1,matrix_transport_pipe2
##### RR in irradiation #####

##### pulse irradiation scenario #####

### JSI TRIGA pulse ###
def process_single_flow_rate_irradiation_scenario(flow_rate, Nmoves, ID_TRIGA, freq_TRIGA, MA2_TRIGA, voxel_volume, output_file_prefix, index):
    """
    Process irradiation scenario for a single flow rate, with unique output based on index.
    """
    dt_TRIGA = 1 / freq_TRIGA
    stop_TRIGA = 400000 * dt_TRIGA
    start_TRIGA = 0
    t = np.arange(start_TRIGA, stop_TRIGA, dt_TRIGA)
    
    # Load pulse data
    pulse_data_TRIGA = pd.read_csv("pulse" + str(ID_TRIGA) + ".txt", names=['Power', 'Temp1', 'Temp2', 'D'], delimiter="\s+")
    M = pulse_data_TRIGA.Power.iloc[0:1000].mean()
    pulse_data_TRIGA.Power = pulse_data_TRIGA.Power - M
    P_MA2_TRIGA = pulse_data_TRIGA.Power.rolling(MA2_TRIGA).mean()
    #P_MA2_TRIGA = pulse_data_TRIGA.Power
    
    # Plot the original pulse and moving average
    plt.figure(1)
    plt.scatter(t, pulse_data_TRIGA.Power, label="Pulse ID=" + str(ID_TRIGA))
    plt.scatter(t, P_MA2_TRIGA, label="MA2 (" + str(MA2_TRIGA) + ")")
    plt.legend()
    plt.savefig(f"{output_file_prefix}_pulse_ID_{ID_TRIGA}_process_{index}.png", dpi=600)

    # Initialize storage arrays
    time_irradiation_TEST = np.zeros(Nmoves)
    power_irradiation_TEST = np.zeros(Nmoves)

    # Calculate delta time based on flow rate
    delta_time = voxel_volume / flow_rate  # time step size in seconds
    time_irradiation_TEST = np.linspace(0, Nmoves * delta_time, Nmoves)
        
    # Define time and signal
    time = t
    signal = P_MA2_TRIGA
    T = max(time)
    Ndt = T / delta_time
        
    print(f"Processing index {index} - Flow Rate: {flow_rate}")
    print(f"T= {T}s")
    print(f"N dt= {int(Ndt)}")
        
    # Define the new time array and resample signal
    new_time = np.linspace(0, T, int(Ndt))
    new_signal_means, new_time_centers = resample_signal_mean_with_new_time(time, signal, new_time)
        
    # Convert to DataFrames for concatenation
    new_signal_means_row = new_signal_means.reshape(1, -1)
    df_time_irradiation_TEST = pd.DataFrame([time_irradiation_TEST])
    df_new_signal = pd.DataFrame(new_signal_means_row)
        
    # Concatenate and fill NaNs with 0
    combined_df = pd.concat([df_time_irradiation_TEST, df_new_signal], ignore_index=True)
    combined_df.fillna(0, inplace=True)
    #print(combined_df)

    # Save and plot results
    plt.figure(2, figsize=(10, 6))
    plt.plot(time, signal, label='Original Signal')
    plt.plot(new_time_centers, new_signal_means, 'o-', label='Mean per New Interval', color='orange')
    plt.xlabel('Time [s]')
    plt.ylabel('Power [MW]')
    plt.legend()
    plt.title('Mean Signal per Interval on New Time Scale')
    plt.savefig(f"{output_file_prefix}_pulse_ID_{ID_TRIGA}_to_KATANA_irradiation_{index}.png", dpi=600)

    plt.figure(3, figsize=(10, 6))
    plt.plot(combined_df.iloc[0, :], combined_df.iloc[1, :], 'o-', label='New data', color='red')
    plt.plot(time, signal, label='Original Signal')
    plt.plot(new_time_centers, new_signal_means, 'o-', label='Mean per New Interval', color='orange')
    plt.xlabel('Time [s]')
    plt.ylabel('Power [MW]')
    plt.legend()
    plt.title('Mean Signal per Interval on New Time Scale')
    plt.savefig(f"{output_file_prefix}_pulse_ID_{ID_TRIGA}_to_KATANA_irradiation_scenario_{index}.png", dpi=600)

    #plt.show()
    return combined_df,df_new_signal
# Helper function (assuming this function exists in your code)
def resample_signal_mean_with_new_time(time, signal, new_time):
    new_signal_means = []
    new_time_centers = []
    
    for i in range(len(new_time) - 1):
        start, end = new_time[i], new_time[i + 1]
        mask = (time >= start) & (time < end)
        
        if np.any(mask):
            mean_value = np.mean(signal[mask])
        else:
            mean_value = np.nan
        
        new_signal_means.append(mean_value)
        interval_center = (start + end) / 2
        new_time_centers.append(interval_center)
    
    return np.array(new_signal_means), np.array(new_time_centers)

#### steady state irradiation scenario #####

def steady_state_irradiation_scenario(flow_rate, Nmoves, irradiation_power, voxel_volume):
    irradiation_power_coefficient = irradiation_power/250  # Scale to 250 kW
    delta_time = voxel_volume / flow_rate  # time step size in seconds
    time_irradiation_TEST = np.linspace(0, Nmoves * delta_time, Nmoves)
    power_irradiation_TEST = np.full(Nmoves, irradiation_power_coefficient)
    
    # Convert to DataFrames for concatenation
    df_time_irradiation_TEST = pd.DataFrame([time_irradiation_TEST])
    df_power_irradiation_TEST = pd.DataFrame([power_irradiation_TEST])
    
    # Concatenate
    combined_df = pd.concat([df_time_irradiation_TEST, df_power_irradiation_TEST], ignore_index=True)
    plt.figure(2, figsize=(10, 6))
    plt.plot(time_irradiation_TEST, power_irradiation_TEST, label='Original Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Power Coefficient[250kW]')
    plt.legend()
    plt.title('Mean Signal per Interval on New Time Scale')
    plt.savefig(output_file_destination+"steady_state_KATANA_irradiation.png", dpi=600)

    return combined_df, df_power_irradiation_TEST

##### GIF creation functions #####

def create_activity_gif_2(
    irradiation_power,
    activity_data,
    time_ramp_TEST,
    output_gif="activity_evolution.gif",
    fps=10,
    colormap=cm.viridis,
    time_values=None,
    vmin=None,
    vmax=None,
    log_scale=False,
    cleanup=True
):
    """
    Creates an animated GIF of a time-evolving 6x22 matrix (heatmap style)
    with a vertical bar representing current reactor power.

    Parameters:
        activity_data (ndarray): Shape (T, 6, 22)
        time_ramp_TEST (tuple): (time_array, power_array), both shape (T,)
        irradiation_power (str): Label for the bar (e.g., "Power")
        output_gif (str): Output filename
        fps (int): Frames per second
        colormap: Matplotlib colormap
        log_scale (bool): Use log scale for heatmap
        cleanup (bool): Delete temp images after creating GIF
    """
    assert activity_data.ndim == 3 and activity_data.shape[1:] == (6, 22), \
        "activity_data must be shape (T, 6, 22)"

    num_frames = activity_data.shape[0]
    time_values = time_ramp_TEST[0].squeeze()
    power_array = time_ramp_TEST[1].squeeze()

    print(len(time_values), len(power_array), num_frames)

    assert len(time_values) == num_frames and len(power_array) == num_frames, \
        "Time and power arrays must match number of frames"

    if log_scale:
        activity_data = np.clip(activity_data, a_min=1e-1, a_max=None)

    if vmin is None:
        vmin = max(1e-1, np.min(activity_data))
    if vmax is None:
        vmax = np.max(activity_data)

    frame_dir = "gif_frames"
    os.makedirs(frame_dir, exist_ok=True)
    frame_paths = []

    for t in range(num_frames):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [4, 1]})

        # --- Heatmap ---
        norm = colors.LogNorm(vmin=vmin, vmax=vmax) if log_scale else None
        cax = ax1.imshow(activity_data[t], cmap=colormap, norm=norm, aspect='auto')
        ax1.set_title(f"Measurement snail: $^{{17}}$N @ t = {time_values[t]:.2f} s", fontsize=14)
        ax1.set_xlabel("Y - voxel index")
        ax1.set_ylabel("X - voxel index")
        fig.colorbar(cax, ax=ax1, label="Activity [Bq]" + (" (log)" if log_scale else ""))

        # --- Power bar (just one bar) ---
        ax2.bar([0], [power_array[t]], color='orange', width=0.8)
        ax2.set_ylim(0, np.max(power_array) * 1.1)
        ax2.set_xlim(-1, 1)
        ax2.set_xticks([])
        ax2.set_title("JSI TRIGA Pulse recorder")
        ax2.set_ylabel("Power [MW]")
        ax2.set_xlabel(irradiation_power)

        # Save and close
        plt.tight_layout()
        frame_path = os.path.join(frame_dir, f"frame_{t:04d}.png")
        plt.savefig(frame_path)
        plt.close(fig)
        frame_paths.append(frame_path)

    # Create GIF
    frames = [Image.open(p) for p in frame_paths]
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0
    )

    if cleanup:
        for p in frame_paths:
            os.remove(p)
        os.rmdir(frame_dir)

    print(f"✅ GIF saved to: {output_gif}")

def animate_activity_with_time_line(
    time_arrays,
    activity_arrays,
    flow_rates,
    output_gif="n17_activity_snail.gif",
    fps=10,
    dpi=150
):
    frame_dir = "n17_frames"
    os.makedirs(frame_dir, exist_ok=True)
    frame_paths = []

    num_flows = len(flow_rates)
    num_frames = len(time_arrays[0])  # assumes aligned time vectors

    # Get y-axis limits from all activity data
    y_min = min([min(a[1]) for a in activity_arrays])
    y_max = max([max(a[1]) for a in activity_arrays])

    for t in range(num_frames):
        plt.figure(figsize=(10, 6))

        # Plot all activity curves
        for i in range(num_flows):
            plt.plot(
                time_arrays[i],
                activity_arrays[i][1],
                label=f"{flow_rates[i]/1000:.2f} l/s"
            )

        # Add vertical line showing current time
        current_time = time_arrays[0][t]
        plt.axvline(current_time, color='black', linestyle='--', linewidth=3)

        # Title with time
        plt.title(f"Measurement snail: $^{{17}}$N @ t = {current_time:.2f} s")

        plt.xlabel("t (s)")
        plt.ylabel("$^{17}N$ Activity (Bq)")
        plt.grid(True, linestyle='--')
        plt.legend()
        plt.ylim(y_min * 0.9, y_max * 1.1)
        plt.tight_layout()

        frame_path = os.path.join(frame_dir, f"frame_{t:04d}.png")
        plt.savefig(frame_path, dpi=dpi)
        plt.close()
        frame_paths.append(frame_path)

    # Create GIF
    frames = [Image.open(p) for p in frame_paths]
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0
    )

    # Clean up
    for p in frame_paths:
        os.remove(p)
    os.rmdir(frame_dir)

    print(f"✅ GIF saved as: {output_gif}")

def animate_activity_with_time_line_V2(
    time_arrays,
    activity_arrays,
    flow_rates,
    scatter_data=None,  # (scatter_time, count_rate)
    output_gif="n17_activity_snail_V2.gif",
    fps=10,
    dpi=150
):
    frame_dir = "n17_frames"
    os.makedirs(frame_dir, exist_ok=True)
    frame_paths = []

    num_flows = len(flow_rates)
    num_frames = len(time_arrays[0])  # assumes aligned time vectors

    # Get y-axis limits from all activity data
    y_min = min([min(a[1]) for a in activity_arrays])
    y_max = max([max(a[1]) for a in activity_arrays])

    # Unpack scatter data
    scatter_time = scatter_rate = None
    if scatter_data:
        scatter_time, scatter_rate = scatter_data

    for t in range(num_frames):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot all activity curves
        for i in range(num_flows):
            ax1.plot(
                time_arrays[i],
                activity_arrays[i][1],
                label="Simulation",
                linewidth=3,
            )

        # Vertical time marker
        current_time = time_arrays[0][t]
        ax1.axvline(current_time, color='black', linestyle='--', linewidth=3)

        # Set labels and limits
        ax1.set_title(f"Measurement snail: $^{{17}}$N @ t = {current_time:.2f} s, pulse ID = 774, FLW1 = 0.66 l/s", fontsize=14)
        ax1.set_xlabel("t (s)")
        ax1.set_ylabel("$^{17}N$ Activity [Bq]")
        ax1.set_ylim(y_min * 0.9, y_max * 1.1)
        ax1.grid(True, linestyle='--')
        ax1.legend(loc="upper left", fontsize=12)

        # Add scatter plot on second y-axis
        if scatter_data:
            cr_min = min(scatter_rate)
            cr_max = max(scatter_rate)
            
            ax2 = ax1.twinx()
            # Filter points with time <= current_time
            mask = [scatter_time[i] <= current_time for i in range(len(scatter_time))]
            filtered_time = [scatter_time[i] for i in range(len(scatter_time)) if mask[i]]
            filtered_rate = [scatter_rate[i] for i in range(len(scatter_rate)) if mask[i]]

            ax2.scatter(filtered_time, filtered_rate, color='red', s=30, alpha=0.7, label='Measurement')
            ax2.set_ylabel("$^{3}He$ Count Rate [CPS]", color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim(cr_min * 0.9, cr_max * 1.1)
            ax2.legend(loc="upper right", fontsize=12)
        
        plt.tight_layout()

        frame_path = os.path.join(frame_dir, f"frame_{t:04d}.png")
        plt.savefig(frame_path, dpi=dpi)
        plt.close()
        frame_paths.append(frame_path)

    # Create GIF
    frames = [Image.open(p) for p in frame_paths]
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0
    )

    # Clean up
    for p in frame_paths:
        os.remove(p)
    os.rmdir(frame_dir)

    print(f"✅ GIF saved as: {output_gif}")

def combine_gifs_vertically(gif1_path, gif2_path, output_path="combined.gif", fps=15):
    gif1 = Image.open(gif1_path)
    gif2 = Image.open(gif2_path)

    frames = []
    duration_ms = int(1000 / fps)

    n_frames = min(gif1.n_frames, gif2.n_frames)

    for i in range(n_frames):
        gif1.seek(i)
        gif2.seek(i)

        # Match width
        target_width = min(gif1.width, gif2.width)
        gif1_resized = gif1.resize((target_width, int(gif1.height * target_width / gif1.width)))
        gif2_resized = gif2.resize((target_width, int(gif2.height * target_width / gif2.width)))

        combined_height = gif1_resized.height + gif2_resized.height
        combined_frame = Image.new("RGB", (target_width, combined_height))

        combined_frame.paste(gif1_resized, (0, 0))
        combined_frame.paste(gif2_resized, (0, gif1_resized.height))

        frames.append(combined_frame)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0
    )

    print(f"✅ Combined GIF saved to: {output_path} at {fps} FPS ({duration_ms}ms/frame)")

##################################################
#
output_file_destination = "output_files/" # Replace with desired destination
# Check if the directory exists
if not os.path.exists(output_file_destination):
    # If not, create it
    os.makedirs(output_file_destination)
    print(f"Directory '{output_file_destination}' created.")
else:
    print(f"Directory '{output_file_destination}' already exists.") 
#   
# Define parameters
# Main loop parameters outside the function with index
###### loop parameters #####
flow_rate_ramp = [670,500,400,300,200,100]  # Example flow rates
Nmoves = 9000 #number of deltaT 4 1538 15380
voxel_volume = 21.65  # cm^3
output_file_prefix = "output"  # Replace with desired prefix
# irradiation scenario
irradiation_scenario = "steady" #pulse or steady

if irradiation_scenario == "steady":
    reactor_power = 250 #kW steady state power
    pump_start = 1000 #number of deltaT befor start of the pump pump_start<Nmoves usualy
    pump_stop = 15256 #number of deltaT after start of the pump
    print("Steady state irradiation scenario")
    irradiation_scenario_data, irradiation_power_data = steady_state_irradiation_scenario(flow_rate_ramp[0], Nmoves, reactor_power, voxel_volume)
    #print(irradiation_scenario_data)

if irradiation_scenario == "pulse":
    ID_TRIGA = [774,774,774] #JSI TRIGA pulse ID
    freq_TRIGA = 20000 #Hz DAQ frequency JSI Pulse recorder
    MA2_TRIGA = 150 #moving average window
    pump_start = 1000 #number of deltaT befor start of the pump pump_start<Nmoves usualy
    pump_stop = 15256 #number of deltaT after start of the pump
    print("Pulse irradiation scenario")
    for index, flow_rate in enumerate(flow_rate_ramp):
        irradiation_scenario_data, irradiation_power_data = process_single_flow_rate_irradiation_scenario(flow_rate, Nmoves, ID_TRIGA[index], freq_TRIGA, MA2_TRIGA, voxel_volume, output_file_prefix, index)
        #print(irradiation_scenario_data)



WACT_RR_data = pd.read_excel("WACT_RR.xlsx") #irradiation data
N16_RR_column = WACT_RR_data['N16RR'] #irradiation data N16
N17_RR_column = WACT_RR_data['N17RR'] #irradiation data N17
O19_RR_column = WACT_RR_data['O19RR'] #irradiation data O19
N16_RR = N16_RR_column.values
N17_RR = N17_RR_column.values
O19_RR = O19_RR_column.values

###### loop parameters #####
# flow_rate_ramp = [5,10,20,30,40,50,60,70,80,90,100,125,150,175,200,225,250,275,300,350,400,450,500,550,600,650,680, 1000, 1500, 2000]
#flow_rate_ramp = [50,100,150,200,250,300,350,400,450,500,550,600,650,700]
#flow_rate_ramp = [680]
#flow_rate_ramp = np.linspace(0, 700, 71)
#flow_rate_ramp = np.append(flow_rate_ramp, [800, 900, 1000, 1100, 1200, 1300, 1400, 1500])

#flow_rate_ramp = [630]  # cm^3 N16_max
#flow_rate_ramp = [630]  # cm^3 N17_max
#flow_rate_ramp = [0.1]  # cm^3 O19_max



header = ["Flow", "s_Activity_N16","s_Activity_N17","s_Activity_O19"]
saturation_activity = pd.DataFrame(columns = header)



activity_measurement_snail_TEST = np.zeros((len(flow_rate_ramp),3,Nmoves)) # storing activity of the measurement snail for every deltaT and every flow rate
time_ramp_TEST = np.zeros((len(flow_rate_ramp),Nmoves)) # storing time for every flow

time_irradiation_TEST = np.zeros((len(flow_rate_ramp),Nmoves)) # storing time for every flow to construct the irradiation scenario
for q in range(len(flow_rate_ramp)):
    flw = flow_rate_ramp[q]
    print("Flow rate:"+str(flw/1000)+" l/s")
    voxel_volume = 21.65 #cm^3
    flow_rate = flw #cm^3/s 
    delta_time = voxel_volume/flow_rate #s
    #    CONFIGURATION_1
    n_voxels_pipe1 = 31 #from irradiation snail to measurement snail
    n_voxels_pipe2 = 49 + 240 #from measurement snail to irradiation snail (pipes + pump) 240 voxels approximatly 5,2l 185 voxels approximatly (4l-->70%)

    #    CONFIGURATION_2
    #n_voxels_pipe1 = 1069 #from irradiation snail to measurement snail
    #n_voxels_pipe2 = 58 + 185 #from measurement snail to irradiation snail (pipes + pump)

    transport_pipe_volume_1 = n_voxels_pipe1*voxel_volume
    transport_pipe_volume_2 = n_voxels_pipe2*voxel_volume
    transport_time_1=transport_pipe_volume_1/flow_rate
    transport_time_2=transport_pipe_volume_2/flow_rate


    t12N16 = 7.13 #s
    lambdaN16 = math.log(2)/t12N16 
    t12N17 = 4.17 #s
    lambdaN17 = math.log(2)/t12N17
    t12O19 = 26.88 #s
    lambdaO19 = math.log(2)/t12O19

    ##### Simulation parameters ###
    time = Nmoves*delta_time
    #pump_start = 1 #number of deltaT befor start of the pump pump_start<Nmoves usualy
    #pump_stop = 15256 #number of deltaT after start of the pump
    Nnuclides=3 #number of nuclides used in the simulation
    nuclides_names = ["O-16","O-17","O-18"] #names of the nuclides used in the simulation
    lambdaWdata = [lambdaN16,lambdaN17,lambdaO19]
    RRWdata = [N16_RR,N17_RR,O19_RR]
    activity_measurement_snail = np.zeros((3,Nmoves)) # storing activity of the measurement snail for every deltaT
    activity_measurement_snail_time_result = np.zeros((Nmoves, 6,22)) # storing activity of the measurement snail for every deltaT

    ##### SIMULATION #####
    for o in range(Nnuclides): 
        matrix_transport_pipe1 = np.zeros((1,n_voxels_pipe1))
        matrix_transport_pipe2 = np.zeros((1,n_voxels_pipe2))
        matrix_irradiation_snail = np.zeros((6, 22)) #irradiation snail
        matrix_measurement_snail = np.zeros((6, 22)) #measurement snail
        
        # nuclide selection
        lambdaW = lambdaWdata[o] # lamnda used
        RR = RRWdata[o]
        print("Start of water activation experiment for nuclide: "+nuclides_names[o])


        for f in range(Nmoves):
            
            if f>pump_start and f<pump_stop:
                matrix_irradiation_snail,matrix_measurement_snail,matrix_transport_pipe1,matrix_transport_pipe2 = transport(matrix_irradiation_snail,matrix_measurement_snail,lambdaW,delta_time,n_voxels_pipe1,n_voxels_pipe2,matrix_transport_pipe1,matrix_transport_pipe2)
            if f>pump_stop:
                matrix_irradiation_snail,matrix_measurement_snail,matrix_transport_pipe1,matrix_transport_pipe2 = decay(matrix_irradiation_snail,matrix_measurement_snail,lambdaW,delta_time,n_voxels_pipe1,n_voxels_pipe2,matrix_transport_pipe1,matrix_transport_pipe2)
            matrix_irradiation_snail = irradiation(matrix_irradiation_snail,RR,delta_time,lambdaW,irradiation_scenario_data.iloc[1,f])
            activity_measurement_snail[o][f] = np.sum(matrix_measurement_snail)*voxel_volume #activity of measurement snail *lambdaW
            activity_measurement_snail_TEST[q][o][f] = np.sum(matrix_measurement_snail)*voxel_volume #activity of measurement snail *lambdaW
            if o==1:
                activity_measurement_snail_time_result[f] = matrix_measurement_snail
    time_ramp_TEST[q]=np.linspace(0, Nmoves*delta_time, Nmoves)
    t=np.linspace(0, len(activity_measurement_snail[0])*delta_time, len(activity_measurement_snail[0]))
    """
    plt.figure(5)
    plt.scatter(t,activity_measurement_snail[0],label="N16")
    plt.xlabel("t (s)")
    plt.ylabel("N16 Activity (Bq)")
    plt.title("Measurement snail")
    plt.legend()
    plt.tight_layout()
    #plt.savefig("N16_activity_1.png",dpi=150)

    plt.figure(6)
    plt.scatter(t,activity_measurement_snail[1]/158,label="N17")
    plt.xlabel("t (s)")
    plt.ylabel("N17 Activity (Bq)")
    plt.title("Measurement snail")
    plt.legend()
    plt.tight_layout()
    #plt.savefig("N17_activity_1.png",dpi=150)

    plt.figure(8)
    plt.scatter(t,activity_measurement_snail[2],label="O19")
    plt.xlabel("t (s)")
    plt.ylabel("O19 Activity (Bq)")
    plt.title("Measurement snail")
    plt.legend()
    plt.tight_layout()
    #plt.savefig("O19_activity_1.png",dpi=150)

    #plt.show()
    """
    saturation_activity.loc[len(saturation_activity)]=[flw,activity_measurement_snail[0][-1],activity_measurement_snail[1][-1],activity_measurement_snail[2][-1]]  # simplified notation 
print(saturation_activity)

print(f"time ramp: {time_ramp_TEST}")


##### DATA VISUALIZATION #####
### GIF: Activity evolution over time ###

gif =0 #whether to create a GIF or not
if gif == 1:
    create_activity_gif_2(
        irradiation_power="TRIGA Pulse",
        activity_data=activity_measurement_snail_time_result,
        time_ramp_TEST=(time_ramp_TEST, irradiation_scenario_data.iloc[1].transpose()),
        output_gif="activity_with_power_bar.gif",
        fps=15,
        log_scale=True,
        colormap=cm.plasma
    )

#animate_activity_with_time_line(
#    time_arrays=time_ramp_TEST,
#    activity_arrays=activity_measurement_snail_TEST,
#    flow_rates=flow_rate_ramp,
#    output_gif="n17_activity_snail.gif",
#    fps=15
#)

    # Suppose you have:
    data_He3 = pd.read_csv("KATANA_FLW1_066ls_TRIGA_rho_200_test_pulse_n_774.csv")
    # Adjust time reference
    data_He3["Time"] = data_He3["Time"] - 15.2
    # Filter out rows with negative time
    data_He3 = data_He3[data_He3["Time"] >= 0].reset_index(drop=True)
    # Extract scatter plot arrays
    scatter_time = data_He3["Time"]
    count_rate = data_He3["Count_Rate"]*10 #it was measured every 0.1s, so we multiply by 10 to get the count rate in CPS

    # Call the function:
    animate_activity_with_time_line_V2(
        time_arrays=time_ramp_TEST,
        activity_arrays=activity_measurement_snail_TEST,
        flow_rates=flow_rate_ramp,
        scatter_data=(scatter_time, count_rate)
    )
    combine_gifs_vertically(
        gif1_path="activity_with_power_bar.gif",
        gif2_path="n17_activity_snail_V2.gif",
        output_path="combined_15fps_V3.gif",
        fps=15
    )

# -------------------------------------------------------------------------------
# Set default font sizes
plt.rcParams.update({'font.size': 16, 'axes.labelsize': 16, 'axes.titlesize': 18})
# -------------------------------------------------------------------------------

#  PLOT: Saturation activity (pump ramp)
plt.figure(1)
plt.plot(saturation_activity["Flow"]/1000,saturation_activity["s_Activity_N16"],label="$^{16}N$")
plt.plot(saturation_activity["Flow"]/1000,saturation_activity["s_Activity_N17"],label="$^{17}N$")
plt.plot(saturation_activity["Flow"]/1000,saturation_activity["s_Activity_O19"],label="$^{19}O$")
plt.yscale("log")
plt.xlabel("Flow rate (l/s)")
plt.ylabel("Activity (Bq)")
plt.title("Measurement snail")
plt.legend()
plt.tight_layout()
plt.savefig(output_file_destination+"activity_pump_ramp.png",dpi=150)

# ------------------------------------------------------------------------------------

#  PLOT: Activity VS flow
if irradiation_scenario == "pulse":
    plt.figure(5)
    for i in range(len(flow_rate_ramp)):
        plt.plot(time_ramp_TEST[i],activity_measurement_snail_TEST[i][0],label=f"{flow_rate_ramp[i]/1000} l/s")
    plt.xlabel("t (s)")
    plt.ylabel("$^{16}N$ Activity (Bq)")
    plt.title("Measurement snail: $^{16}N$")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_file_destination+"N16_activity_pulse_ID_"+str(ID_TRIGA[1])+".png",dpi=150)

    plt.figure(6)
    for i in range(len(flow_rate_ramp)):
        plt.plot(time_ramp_TEST[i],activity_measurement_snail_TEST[i][1],label=f"{flow_rate_ramp[i]/1000} l/s")
    plt.xlabel("t (s)")
    plt.ylabel("$^{17}N$ Activity (Bq)")
    plt.title("Measurement snail: $^{17}N$")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(+output_file_destination+"N17_activity_pulse_ID_"+str(ID_TRIGA[1])+".png",dpi=150)

    plt.figure(8)
    for i in range(len(flow_rate_ramp)):
        plt.plot(time_ramp_TEST[i],activity_measurement_snail_TEST[i][1],label=f"{flow_rate_ramp[i]/1000} l/s")
    plt.xlabel("t (s)")
    plt.ylabel("$^{17}N$ Activity (Bq)")
    plt.title("Measurement snail: $^{17}N$")
    plt.legend()
    plt.xlim(0,50)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_file_destination+"N17_activity_pulse_ID_"+str(ID_TRIGA[1])+"_zoom.png",dpi=150)

    plt.figure(7)
    for i in range(len(flow_rate_ramp)):
        plt.plot(time_ramp_TEST[i],activity_measurement_snail_TEST[i][2],label=f"{flow_rate_ramp[i]/1000} l/s")
    plt.xlabel("t (s)")
    plt.ylabel("$^{19}O$ Activity (Bq)")
    plt.title("Measurement snail: $^{19}O$")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_file_destination+"O19_activity_pulse_ID_"+str(ID_TRIGA[1])+".png",dpi=150)
else:
    plt.figure(5)
    for i in range(len(flow_rate_ramp)):
        plt.plot(time_ramp_TEST[i],activity_measurement_snail_TEST[i][0],label=f"{flow_rate_ramp[i]/1000} l/s")
    plt.xlabel("t (s)")
    plt.ylabel("$^{16}N$ Activity (Bq)")
    plt.title("Measurement snail: $^{16}N$")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_file_destination+"N16_activity_flw1_"+str(flow_rate_ramp[0])+".png",dpi=150)

    plt.figure(6)
    for i in range(len(flow_rate_ramp)):
        plt.plot(time_ramp_TEST[i],activity_measurement_snail_TEST[i][1],label=f"{flow_rate_ramp[i]/1000} l/s")
    plt.xlabel("t (s)")
    plt.ylabel("$^{17}N$ Activity (Bq)")
    plt.title("Measurement snail: $^{17}N$")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_file_destination+"N17_activity_flw1_"+str(flow_rate_ramp[0])+".png",dpi=150)

    plt.figure(8)
    for i in range(len(flow_rate_ramp)):
        plt.plot(time_ramp_TEST[i],activity_measurement_snail_TEST[i][1],label=f"{flow_rate_ramp[i]/1000} l/s")
    plt.xlabel("t (s)")
    plt.ylabel("$^{17}N$ Activity (Bq)")
    plt.title("Measurement snail: $^{17}N$")
    plt.legend()
    plt.xlim(0,50)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_file_destination+"N17_activity_flw1_"+str(flow_rate_ramp[0])+"_zoom.png",dpi=150)

    plt.figure(7)
    for i in range(len(flow_rate_ramp)):
        plt.plot(time_ramp_TEST[i],activity_measurement_snail_TEST[i][2],label=f"{flow_rate_ramp[i]/1000} l/s")
    plt.xlabel("t (s)")
    plt.ylabel("$^{19}O$ Activity (Bq)")
    plt.title("Measurement snail: $^{19}O$")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_file_destination+"/O19_activity_flw1_"+str(flow_rate_ramp[0])+".png",dpi=150)

# ---------------------------------------------------------------------------------
# WRITE TO EXCEL AT THE END OF THE SCRIPT
# ---------------------------------------------------------------------------------
# SATURATION ACTIVITY

# ------------------------------------------------------------------------------------
print("Simulation completed.")
plt.show()

