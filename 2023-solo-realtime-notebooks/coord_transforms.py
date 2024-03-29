import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import itertools


#input datetime to return T1, T2 and T3 based on Hapgood 1992
#http://www.igpp.ucla.edu/public/vassilis/ESS261/Lecture03/Hapgood_sdarticle.pdf
def get_transformation_matrices(timestamp):
    #format dates correctly, calculate MJD, T0, UT 
    ts = pd.Timestamp(timestamp)
    jd=ts.to_julian_date()
    mjd=float(int(jd-2400000.5)) #use modified julian date    
    T0=(mjd-51544.5)/36525.0
    UT=ts.hour + ts.minute / 60. + ts.second / 3600. #time in UT in hours    
    #define position of geomagnetic pole in GEO coordinates
    pgeo=78.8+4.283*((mjd-46066)/365.25)*0.01 #in degrees
    lgeo=289.1-1.413*((mjd-46066)/365.25)*0.01 #in degrees
    #GEO vector
    Qg=[np.cos(pgeo*np.pi/180)*np.cos(lgeo*np.pi/180), np.cos(pgeo*np.pi/180)*np.sin(lgeo*np.pi/180), np.sin(pgeo*np.pi/180)]
    #now move to equation at the end of the section, which goes back to equations 2 and 4:
    #CREATE T1; T0, UT is known from above
    zeta=(100.461+36000.770*T0+15.04107*UT)*np.pi/180
    ################### theta und z
    T1=np.matrix([[np.cos(zeta), np.sin(zeta),  0], [-np.sin(zeta) , np.cos(zeta) , 0], [0,  0,  1]]) #angle for transpose
    LAMBDA=280.460+36000.772*T0+0.04107*UT
    M=357.528+35999.050*T0+0.04107*UT
    lt2=(LAMBDA+(1.915-0.0048*T0)*np.sin(M*np.pi/180)+0.020*np.sin(2*M*np.pi/180))*np.pi/180
    #CREATE T2, LAMBDA, M, lt2 known from above
    ##################### lamdbda und Z
    t2z=np.matrix([[np.cos(lt2), np.sin(lt2),  0], [-np.sin(lt2) , np.cos(lt2) , 0], [0,  0,  1]])
    et2=(23.439-0.013*T0)*np.pi/180
    ###################### epsilon und x
    t2x=np.matrix([[1,0,0],[0,np.cos(et2), np.sin(et2)], [0, -np.sin(et2), np.cos(et2)]])
    T2=np.dot(t2z,t2x)  #equation 4 in Hapgood 1992
    #matrix multiplications   
    T2T1t=np.dot(T2,np.matrix.transpose(T1))
    ################
    Qe=np.dot(T2T1t,Qg) #Q=T2*T1^-1*Qq
    psigsm=np.arctan(Qe.item(1)/Qe.item(2)) #arctan(ye/ze) in between -pi/2 to +pi/2
    T3=np.matrix([[1,0,0],[0,np.cos(-psigsm), np.sin(-psigsm)], [0, -np.sin(-psigsm), np.cos(-psigsm)]])
    return T1, T2, T3


def GSE_to_GSM(df):
    B_GSM = []
    for i in range(df.shape[0]):
        T1, T2, T3 = get_transformation_matrices(df['timestamp'].iloc[0])
        B_GSE_i = np.matrix([[df['b_x'].iloc[i]],[df['b_y'].iloc[i]],[df['b_z'].iloc[i]]]) 
        B_GSM_i = np.dot(T3,B_GSE_i)
        B_GSM_i_list = B_GSM_i.tolist()
        flat_B_GSM_i = list(itertools.chain(*B_GSM_i_list))
        B_GSM.append(flat_B_GSM_i)
    df_transformed = pd.DataFrame(B_GSM, columns=['b_x', 'b_y', 'b_z'])
    df_transformed['b_tot'] = np.linalg.norm(df_transformed[['b_x', 'b_y', 'b_z']], axis=1)
    df_transformed['timestamp'] = df['timestamp']
    return df_transformed


def GSM_to_GSE(df):
    B_GSE = []
    for i in range(df.shape[0]):
        T1, T2, T3 = get_transformation_matrices(df['timestamp'].iloc[0])
        T3_inv = np.linalg.inv(T3)
        B_GSM_i = np.matrix([[df['b_x'].iloc[i]],[df['b_y'].iloc[i]],[df['b_z'].iloc[i]]]) 
        B_GSE_i = np.dot(T3_inv,B_GSM_i)
        B_GSE_i_list = B_GSE_i.tolist()
        flat_B_GSE_i = list(itertools.chain(*B_GSE_i_list))
        B_GSE.append(flat_B_GSE_i)
    df_transformed = pd.DataFrame(B_GSE, columns=['b_x', 'b_y', 'b_z'])
    df_transformed['b_tot'] = np.linalg.norm(df_transformed[['b_x', 'b_y', 'b_z']], axis=1)
    df_transformed['timestamp'] = df['timestamp']
    return df_transformed


def GSE_to_RTN_approx(df):
    df_transformed = pd.DataFrame()
    df_transformed['timestamp'] = df['timestamp']
    df_transformed['b_tot'] = df['b_tot']
    df_transformed['b_x'] = -df['b_x']
    df_transformed['b_y'] = -df['b_y']
    df_transformed['b_z'] = df['b_z']
    return df_transformed

    
def GSM_to_RTN_approx(df):
    df_gse = GSM_to_GSE(df)
    df_transformed = GSE_to_RTN_approx(df_gse)
    return df_transformed