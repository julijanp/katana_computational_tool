#Author: Julijan Peric
#Project: KATANA - water activation model - pulse operation
#Version: 1 (KATANA pulse paper) -> from V7
#ToDo:
#Pump volume --> 185 voxels approximatly (4l-->70%)

##################################################################
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


# funtions
#irradiation in irradiation snail
def irradiation(matrix_irradiation_snail,N16_RR,delta_time,lambdaN16):
    for j in range(6):
        for i in range(22):
            matrix_irradiation_snail[j][i]=(matrix_irradiation_snail[j][i]*math.exp(-lambdaN16*delta_time)+(N16_RR[i]*(1-math.exp(-lambdaN16*delta_time))))
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
    matrix_transport_pipe2t1[0][n_voxels_pipe2-1]=matrix_measurement_snail[5][0]*math.exp(-lambdaN16*delta_time) #from measurement volume last voxel to irradiation volume

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

WACT_RR_data = pd.read_excel("WACT_RR.xlsx") #irradiation data
N16_RR_column = WACT_RR_data['N16RR'] #irradiation data N16
N17_RR_column = WACT_RR_data['N17RR'] #irradiation data N17
O19_RR_column = WACT_RR_data['O19RR'] #irradiation data O19
N16_RR = N16_RR_column.values
N17_RR = N17_RR_column.values
O19_RR = O19_RR_column.values

###### loop parameters #####
# flow_rate_ramp = [5,10,20,30,40,50,60,70,80,90,100,125,150,175,200,225,250,275,300,350,400,450,500,550,600,650,680, 1000, 1500, 2000]
flow_rate_ramp = [50,100,150,200,250,300,350,400,450,500,550,600,650,700]
#flow_rate_ramp = [680]
#flow_rate_ramp = np.linspace(0, 700, 71)
#flow_rate_ramp = np.append(flow_rate_ramp, [800, 900, 1000, 1100, 1200, 1300, 1400, 1500])

#flow_rate_ramp = [630]  # cm^3 N16_max
#flow_rate_ramp = [630]  # cm^3 N17_max
#flow_rate_ramp = [0.1]  # cm^3 O19_max

Nmoves = 15384 #number of deltaT --> time 10000

header = ["Flow", "s_Activity_N16","s_Activity_N17","s_Activity_O19"]
saturation_activity = pd.DataFrame(columns = header)



activity_measurement_snail_TEST = np.zeros((len(flow_rate_ramp),3,Nmoves)) # storing activity of the measurement snail for every deltaT and every flow rate
time_ramp_TEST = np.zeros((len(flow_rate_ramp),Nmoves)) # storing time for every flow

time_irradiation_TEST = np.zeros((len(flow_rate_ramp),Nmoves)) # storing time for every flow to construct the irradiation scenario

for q in range(len(flow_rate_ramp)):
    flw = flow_rate_ramp[q]
    print(flw)
    voxel_volume = 21.65 #cm^3
    flow_rate = flw #cm^3/s 
    delta_time = voxel_volume/flow_rate #s
    #    CONFIGURATION_1
    n_voxels_pipe1 = 31 #from irradiation snail to measurement snail
    n_voxels_pipe2 = 49 + 185 #from measurement snail to irradiation snail (pipes + pump)

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
    pump_start = 5128 #number of deltaT befor start of the pump pump_start<Nmoves usualy
    pump_stop = 10256 #number of deltaT after start of the pump
    Nnuclides=3 #number of nuclides used in the simulation
    lambdaWdata = [lambdaN16,lambdaN17,lambdaO19]
    RRWdata = [N16_RR,N17_RR,O19_RR]
    activity_measurement_snail = np.zeros((3,Nmoves)) # storing activity of the measurement snail for every deltaT
    

    ##### SIMULATION #####
    for o in range(Nnuclides): 
        matrix_transport_pipe1 = np.zeros((1,n_voxels_pipe1))
        matrix_transport_pipe2 = np.zeros((1,n_voxels_pipe2))
        matrix_irradiation_snail = np.zeros((6, 22)) #irradiation snail
        matrix_measurement_snail = np.zeros((6, 22)) #measurement snail
        
        # nuclide selection
        lambdaW = lambdaWdata[o] # lamnda used
        RR = RRWdata[o]
        print("Start of water activation experiment")


        for f in range(Nmoves):
            
            if f>pump_start and f<pump_stop:
                matrix_irradiation_snail,matrix_measurement_snail,matrix_transport_pipe1,matrix_transport_pipe2 = transport(matrix_irradiation_snail,matrix_measurement_snail,lambdaW,delta_time,n_voxels_pipe1,n_voxels_pipe2,matrix_transport_pipe1,matrix_transport_pipe2)
            if f>pump_stop:
                matrix_irradiation_snail,matrix_measurement_snail,matrix_transport_pipe1,matrix_transport_pipe2 = decay(matrix_irradiation_snail,matrix_measurement_snail,lambdaW,delta_time,n_voxels_pipe1,n_voxels_pipe2,matrix_transport_pipe1,matrix_transport_pipe2)
            matrix_irradiation_snail = irradiation(matrix_irradiation_snail,RR,delta_time,lambdaW)
            activity_measurement_snail[o][f] = np.sum(matrix_measurement_snail)*voxel_volume #activity of measurement snail *lambdaW
            activity_measurement_snail_TEST[q][o][f] = np.sum(matrix_measurement_snail)*voxel_volume #activity of measurement snail *lambdaW
    
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
plt.savefig("activity_pump_ramp.png",dpi=150)

# ------------------------------------------------------------------------------------

#  PLOT: Activity VS flow
plt.figure(5)
for i in range(len(flow_rate_ramp)):
    plt.plot(time_ramp_TEST[i],activity_measurement_snail_TEST[i][0],label=f"{flow_rate_ramp[i]/1000} l/s")
plt.xlabel("t (s)")
plt.ylabel("$^{16}N$ Activity (Bq)")
plt.title("Measurement snail: $^{16}N$")
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
#plt.savefig("N16_activity_1.png",dpi=150)

plt.figure(6)
for i in range(len(flow_rate_ramp)):
    plt.plot(time_ramp_TEST[i],activity_measurement_snail_TEST[i][1],label=f"{flow_rate_ramp[i]/1000} l/s")
plt.xlabel("t (s)")
plt.ylabel("$^{17}N$ Activity (Bq)")
plt.title("Measurement snail: $^{17}N$")
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
#plt.savefig("N17_activity_1.png",dpi=150)

plt.figure(7)
for i in range(len(flow_rate_ramp)):
    plt.plot(time_ramp_TEST[i],activity_measurement_snail_TEST[i][2],label=f"{flow_rate_ramp[i]/1000} l/s")
plt.xlabel("t (s)")
plt.ylabel("$^{19}O$ Activity (Bq)")
plt.title("Measurement snail: $^{19}O$")
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
#plt.savefig("O19_activity_1.png",dpi=150)


# ---------------------------------------------------------------------------------
# WRITE TO EXCEL AT THE END OF THE SCRIPT
# ---------------------------------------------------------------------------------
# SATURATION ACTIVITY

#Sat_Activity_output_file_path = "Saturation_activity-VS-flow_rate.xlsx"  # Specify your desired file path
#saturation_activity.to_excel(Sat_Activity_output_file_path, index=False)

# ------------------------------------------------------------------------------------
# EXCEL: ACTIVITY VS FLOW 

# Create a DataFrame to store the data
cases = ['N16', 'N17', 'O19']

for j in range(len(cases)):  # Loop over three different indices (you can adjust the range as needed)
    data = {f'{label}_{flow_rate/1000}_l/s': [] for flow_rate in flow_rate_ramp for label in ['Time_(s)', 'Activity']}

    for i in range(len(flow_rate_ramp)):
        time_i = time_ramp_TEST[i]
        activity_i = activity_measurement_snail_TEST[i][j]

        if len(time_i) != len(activity_i):
            raise ValueError(f"Length mismatch at index {i}, activity index {j}: 'Time' has length {len(time_i)}, 'Activity' has length {len(activity_i)}")

        data[f'Time_(s)_{flow_rate_ramp[i]/1000}_l/s'].extend(time_i)
        data[f'Activity_{flow_rate_ramp[i]/1000}_l/s'].extend(activity_i)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to an Excel file
    output_file_path = f'Activity_vs_Flow_{cases[j]}.xlsx'
    df.to_excel(output_file_path, index=False)

    print(f"Data for {cases[j]}: {data}")
    print(f"DataFrame saved to {output_file_path}")         
    data.clear()
# ---------------------------------------------------------------------------------


plt.show()

