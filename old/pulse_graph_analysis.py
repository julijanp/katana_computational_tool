import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import peak_widths


pulse_data_id_1=pd.read_excel("triga_pulses_1.xlsx")
#print(pulse_data_id_1)
freq=20000 #hz
dt=1/freq
stop=400000*dt

pulse_prop=pd.DataFrame(columns=['ID','P_P_R','FWHM_R','ER_R','MA1','P_P_1','FWHM_1','ER_1','MA2','P_P_2','FWHM_2','ER_2'])



start=0
i=0
t=np.arange(start,stop,dt)
for ID in pulse_data_id_1.iloc[:,0]:
    ID=ID
    print("Pulse ID: "+str(ID))
#ID=568

    # zero adjustment---------------------------------------------------------------------------------------
    pulse_data = pd.read_csv("pulse"+str(ID)+".txt",names=['Power', 'Temp1', 'Temp2','D'], delimiter="\s+")
    M=pulse_data.Power.iloc[0:1000].mean()
    #print(M)
    #-------------------------------------------------------------------------------------------------------
    pulse_data.Power=pulse_data.Power-M
    #print(pulse_data)
    MA1=15 #moving average window
    P_MA1=pulse_data.Power.rolling(MA1).mean()
    #P_MA1=P_MA1.fillna(value=0)
    #print(P_MA1)
    MA2=150 #moving average window
    P_MA2=pulse_data.Power.rolling(MA2).mean()

    # pulse properties calculation
    #! raw signal
    maximum_raw=np.nanmax(pulse_data.Power) #maximum of signal
    maximumind_raw=np.argmax(pulse_data.Power) #index of maximum of signal
    maximuminda_raw=np.array([maximumind_raw])
    fwhm_raw= peak_widths(pulse_data.Power,maximuminda_raw,rel_height=0.5) #FWHM
    # print pulse properties

    print("Peak signal:"+str(maximum_raw))
    print(maximumind_raw)
    print("FWHM:"+str(float(fwhm_raw[0]*dt)))

    #integral
    a=1000
    energy_realesed_raw=np.trapz(pulse_data.Power[(maximumind_raw-a):(maximumind_raw+a)],dx=dt)
    print("Released energy: "+str(energy_realesed_raw)+" a.u.")

    #! MA1 signal
    maximum_1=np.nanmax(P_MA1) #maximum of signal
    maximumind_1=np.argmax(P_MA1) #index of maximum of signal
    maximuminda_1=np.array([maximumind_1])
    fwhm_1= peak_widths(P_MA1,maximuminda_1,rel_height=0.5) #FWHM
    # print pulse properties
    print(maximum_1)
    print("Moving average: "+str(MA1))
    print("Peak signal:"+str(maximum_1)+"MW")
    print("FWHM:"+str(float(fwhm_1[0]*dt))+"s")

    #integral
    a=1000
    energy_realesed_1=np.trapz(P_MA1[(maximumind_1-a):(maximumind_1+a)],dx=dt)
    print("Released energy: "+str(energy_realesed_1)+"MWs")

    #! MA2 signal
    maximum_2=np.nanmax(P_MA2) #maximum of signal
    maximumind_2=np.argmax(P_MA2) #index of maximum of signal
    maximuminda_2=np.array([maximumind_2])
    fwhm_2= peak_widths(P_MA2,maximuminda_2,rel_height=0.5) #FWHM
    # print pulse properties
    print("Moving average: "+str(MA2))
    print("Peak signal:"+str(maximum_2)+"MW")
    print("FWHM:"+str(float(fwhm_2[0]*dt))+"s")

    #integral
    a=1000
    energy_realesed_2=np.trapz(P_MA2[(maximumind_2-a):(maximumind_2+a)],dx=dt)
    print("Released energy: "+str(energy_realesed_2)+"MWs")


    # new row add -------------------------------------------------------------------------------------
    new_row =pd.Series({'ID': ID,
        'P_P_R': maximum_raw,
        'FWHM_R': float(fwhm_raw[0]*dt),
        'ER_R': energy_realesed_raw,
        'MA1': MA1,
        'P_P_1': maximum_1,
        'FWHM_1': float(fwhm_1[0]*dt),
        'ER_1': energy_realesed_1,
        'MA2': MA2,
        'P_P_2': maximum_2,
        'FWHM_2': float(fwhm_2[0]*dt),
        'ER_2': energy_realesed_2,})

    pulse_prop=pd.concat([pulse_prop,new_row.to_frame().T], ignore_index=True)
    print(pulse_prop)
    #---------------------------------------------------------------------------------------------------

    # draw to inspect
        

    plt.figure(1)
    plt.scatter(t,pulse_data.Power,label="Pulse ID="+str(ID)+", \u03C1="+str(pulse_data_id_1.iloc[i,1])+"$")
    plt.scatter(t,P_MA1,label="MA1 ("+str(MA1)+")")
    plt.scatter(t,P_MA2,label="MA1 ("+str(MA2)+")")
    #raw signal
    plt.scatter(t[maximumind_raw],pulse_data.Power[maximumind_raw], color="Blue",label="Signal peak - Raw") 
    plt.hlines(fwhm_raw[1],fwhm_raw[2]*dt,fwhm_raw[3]*dt, color="Blue", linewidth=5, label="FWHM - Raw")
    #MA 1
    plt.scatter(t[maximumind_1],P_MA1[maximumind_1], color="Orangered",label="Signal peak - MA1") 
    plt.hlines(fwhm_1[1],fwhm_1[2]*dt,fwhm_1[3]*dt, color="Orangered", linewidth=5, label="FWHM - MA1")

    #MA 2
    plt.scatter(t[maximumind_2],P_MA2[maximumind_2], color="Limegreen",label="Signal peak - MA2") 
    plt.hlines(fwhm_2[1],fwhm_2[2]*dt,fwhm_2[3]*dt, color="Limegreen", linewidth=5, label="FWHM - MA2")

    plt.xlabel("t [s]")
    plt.ylabel("P [MW]")
    plt.title("TRIGA Pulse Recorder")
    plt.legend()
    plt.savefig("triga_pulse_proccesing_"+str(ID)+".png", bbox_inches='tight')
    plt.clf()
    
    plt.figure(2)
    plt.scatter(t,pulse_data.Power,color="C0",label="Pulse ID="+str(ID)+", \u03C1="+str(pulse_data_id_1.iloc[i,1])+"$")
    #plt.scatter(t,P_MA1,label="MA1 ("+str(MA1)+")")
    #plt.scatter(t,P_MA2,label="MA1 ("+str(MA2)+")")
    #raw signal
    plt.scatter(t[maximumind_raw],pulse_data.Power[maximumind_raw],color="C1",label="Signal peak - Raw") 
    plt.hlines(fwhm_raw[1],fwhm_raw[2]*dt,fwhm_raw[3]*dt, linewidth=5,color="C2",label="FWHM - Raw")
    #MA 1
    #plt.scatter(t[maximumind_1],P_MA1[maximumind_1], color="Orangered",label="Signal peak - MA1") 
    #plt.hlines(fwhm_1[1],fwhm_1[2]*dt,fwhm_1[3]*dt, color="Orangered", linewidth=5, label="FWHM - MA1")

    #MA 2
    #plt.scatter(t[maximumind_2],P_MA2[maximumind_2], color="Limegreen",label="Signal peak - MA1") 
    #plt.hlines(fwhm_2[1],fwhm_2[2]*dt,fwhm_2[3]*dt, color="Limegreen", linewidth=5, label="FWHM - MA1")

    plt.xlabel("t [s]")
    plt.ylabel("P [MW]")
    plt.title("TRIGA Pulse Recorder")
    plt.legend()
    plt.savefig("triga_pulse_proccesing_raw_"+str(ID)+".png", bbox_inches='tight')
    plt.clf()
    
    plt.figure(3)
    #plt.scatter(t,pulse_data.Power,label="Pulse ID="+str(ID)+", \u03C1="+str(pulse_data_id_1.iloc[i,1])+"$")
    plt.scatter(t,P_MA1,color="C0",label="MA1 ("+str(MA1)+")")
    #plt.scatter(t,P_MA2,label="MA1 ("+str(MA2)+")")
    #raw signal
    #plt.scatter(t[maximumind_raw],pulse_data.Power[maximumind_raw], color="Blue",label="Signal peak - Raw") 
    #plt.hlines(fwhm_raw[1],fwhm_raw[2]*dt,fwhm_raw[3]*dt, color="Blue", linewidth=5, label="FWHM - Raw")
    #MA 1
    plt.scatter(t[maximumind_1],P_MA1[maximumind_1],color="C1",label="Signal peak - MA1") 
    plt.hlines(fwhm_1[1],fwhm_1[2]*dt,fwhm_1[3]*dt,color="C2", linewidth=5, label="FWHM - MA1")

    #MA 2
    #plt.scatter(t[maximumind_2],P_MA2[maximumind_2], color="Limegreen",label="Signal peak - MA1") 
    #plt.hlines(fwhm_2[1],fwhm_2[2]*dt,fwhm_2[3]*dt, color="Limegreen", linewidth=5, label="FWHM - MA1")

    plt.xlabel("t [s]")
    plt.ylabel("P [MW]")
    plt.title("TRIGA Pulse Recorder")
    plt.legend()
    plt.savefig("triga_pulse_proccesing_MA1_"+str(ID)+".png", bbox_inches='tight')
    plt.clf()
    
    plt.figure(4)
    #plt.scatter(t,pulse_data.Power,label="Pulse ID="+str(ID)+", \u03C1="+str(pulse_data_id_1.iloc[i,1])+"$")
    #plt.scatter(t,P_MA1,label="MA1 ("+str(MA1)+")")
    plt.scatter(t,P_MA2,color="C0",label="MA1 ("+str(MA2)+")")
    #raw signal
    #plt.scatter(t[maximumind_raw],pulse_data.Power[maximumind_raw], color="Blue",label="Signal peak - Raw") 
    #plt.hlines(fwhm_raw[1],fwhm_raw[2]*dt,fwhm_raw[3]*dt, color="Blue", linewidth=5, label="FWHM - Raw")
    #MA 1
    #plt.scatter(t[maximumind_1],P_MA1[maximumind_1], color="Orangered",label="Signal peak - MA1") 
    #plt.hlines(fwhm_1[1],fwhm_1[2]*dt,fwhm_1[3]*dt, color="Orangered", linewidth=5, label="FWHM - MA1")

    #MA 2
    plt.scatter(t[maximumind_2],P_MA2[maximumind_2],color="C1",label="Signal peak - MA2") 
    plt.hlines(fwhm_2[1],fwhm_2[2]*dt,fwhm_2[3]*dt,color="C2", linewidth=5, label="FWHM - MA2")

    plt.xlabel("t [s]")
    plt.ylabel("P [MW]")
    plt.title("TRIGA Pulse Recorder")
    plt.legend()
    plt.savefig("triga_pulse_proccesing_MA2_"+str(ID)+".png", bbox_inches='tight')
    plt.clf()


    #plt.figure(2)
    #plt.scatter(t,pulse_data.Temp1,label="Pulse ID="+str(ID)+", \u03C1="+str(pulse_data_id_1.iloc[i,1])+"$")
    #plt.xlabel("t [s]")
    #plt.ylabel("T [°C]")
    #plt.title("TRIGA Pulse Recorder")
    #plt.legend()



    #plt.show()


#plt.savefig("triga_pulse_"+str(ID)+".png", bbox_inches='tight')
#plt.clf()
pulse_prop.to_excel('pulse_TRIGA_properties.xlsx')

    
