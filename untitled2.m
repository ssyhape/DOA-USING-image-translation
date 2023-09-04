 P = arr;
 ap_pos = [3.961,0;3.987,0;4.013,0;4.039,0];
 theta_vals  = -90:1:90;
 d_vals = 0:0.02:11;
 d1 = 0:0.02:8;
 d2 = 0:0.02:5;
 P_out=convert_multipathProfile_to_xy(P,theta_vals,d_vals,d1,d2,ap_pos);

