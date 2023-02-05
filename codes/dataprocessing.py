import pandas as pd



def preprocessing(df):

    laser_lis = []
    for n in range(1, 13):
        laser = df.iloc[:, 90*(n-1):90*n].mean(axis=1)
        laser_lis.append(laser)

    laser_lis

    laser_1 = laser_lis[0]
    laser_2 = laser_lis[1]
    laser_3 = laser_lis[2]
    laser_4 = laser_lis[3]
    laser_5 = laser_lis[4]
    laser_6 = laser_lis[5]
    laser_7 = laser_lis[6]
    laser_8 = laser_lis[7]
    laser_9 = laser_lis[8]
    laser_10 = laser_lis[9]
    laser_11 = laser_lis[10]
    laser_12 = laser_lis[11]
    
    goal_xf = df.iloc[:, 1080]
    goal_yf = df.iloc[:, 1081]
    goal_qkf = df.iloc[:, 1082]
    goal_qrf = df.iloc[:, 1083]
    goal_xl = df.iloc[:, 1084]
    goal_yl = df.iloc[:, 1085]
    goal_qkl = df.iloc[:, 1086]
    goal_qrl = df.iloc[:, 1087]
    pos_x = df.iloc[:, 1088]
    pos_y = df.iloc[:, 1089]
    pos_qk = df.iloc[:, 1090]
    pos_qr = df.iloc[:, 1091]
    pos_qr = df.iloc[:, 1091]

    rel_xg = goal_xf - pos_x
    rel_yg = goal_yf - pos_y
    rel_xl = goal_xl - pos_x
    rel_yl = goal_yl - pos_x
    rel_qkg = goal_qkf- pos_qk
    rel_qrg = goal_qrf- pos_qr
    rel_qkl = goal_qkl- pos_qk
    rel_qrl = goal_qrl- pos_qr

    cmd_vel_v = df.iloc[:, 1092]
    cmd_vel_w = df.iloc[:, 1093]

    X =  pd.concat([laser_1,laser_2,laser_3,laser_4,laser_5,laser_6,laser_7,laser_8,laser_9,laser_10,laser_11,laser_12,rel_xg,rel_yg,rel_xl,rel_yl,rel_qkg,rel_qrg,rel_qkl,rel_qrl],axis=1)
    Y =  pd.concat([cmd_vel_v, cmd_vel_w],axis=1)
    T = pd.concat([X, Y], axis = 1)

    return X, Y , T
