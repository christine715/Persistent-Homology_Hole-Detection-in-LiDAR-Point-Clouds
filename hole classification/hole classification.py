import numpy as np
from numpy.linalg import inv
import math

#unknown data
#Experiment
unknown_LDis=22.06
unknown_3darea=390.48
unknown_STDZ=3.43


unknown_data=np.array([[unknown_3darea],[unknown_LDis],[unknown_STDZ]])


#input train data
no_feature_set=3

#########################Occlusion


input_occlusion_3dArea=r"E:\fyp on ccm computer\Maximum Likelihood Estimation\Maximum Likelihood Estimation\occu\3dArea.txt"
occlusion_3dArea = np.loadtxt(input_occlusion_3dArea)
occlusion_3dArea=occlusion_3dArea.astype(float)

input_occlusion_LDis=r"E:\fyp on ccm computer\Maximum Likelihood Estimation\Maximum Likelihood Estimation\occu\LDis.txt"
occlusion_LDis= np.loadtxt(input_occlusion_LDis)

input_occlusion_STDZ=r"E:\fyp on ccm computer\Maximum Likelihood Estimation\Maximum Likelihood Estimation\occu\std_Z.txt"
occlusion_STDZ= np.loadtxt(input_occlusion_STDZ)

#cal mean
mean_occlusion_3dArea=np.mean(occlusion_3dArea)
mean_occlusion_LDis=np.mean(occlusion_LDis)
mean_occlusion_STDZ=np.mean(occlusion_STDZ)
#cal variance
var_occlusion_3dArea=np.var(occlusion_3dArea,ddof=1)
var_occlusion_LDis=np.var(occlusion_LDis,ddof=1)
var_occlusion_STDZ=np.var(occlusion_STDZ,ddof=1)
#cal mean matrix
mean_matrix_occlusion=np.array([[mean_occlusion_3dArea],[mean_occlusion_LDis],[mean_occlusion_STDZ]])
#cal cvc matrix
vcv_matrix_occlusion=np.cov(np.array([occlusion_3dArea,occlusion_LDis,occlusion_STDZ]))

#cal probability x in a
transposeX_M_occlusion=(unknown_data-mean_matrix_occlusion).transpose() #[X-M]^T
inverse_vcv_occlusion=inv(vcv_matrix_occlusion)#V^-1
Times_occlusion=(transposeX_M_occlusion).dot(inverse_vcv_occlusion) #[X-M]^T*V^-1
Times_data_occlusion=Times_occlusion.dot(unknown_data-mean_matrix_occlusion)#[X-M]^T*V^-1*[X-M]
Times_data_occlusion = Times_data_occlusion[0]
tempPX_occlusion=(1/2)*Times_data_occlusion#(1/2)*[X-M]^T*V^-1*[X-M]
det_vcv_occlusion=np.linalg.det(vcv_matrix_occlusion)#|V|
log_det_vcv=math.log(det_vcv_occlusion)#loge|V|
log_times_e=(-1/2)*log_det_vcv#(-1/2)*loge|V|
PX_occlusion=log_times_e-tempPX_occlusion#(-1/2)*loge|V|-(1/2)*[X-M]^T*V^-1*[X-M]
print(PX_occlusion)

#####################Rooftop
# input_roof_2dArea=r"E:\fyp on ccm computer\Maximum Likelihood Estimation\Maximum Likelihood Estimation\roof\2dArea.txt"
# roof_2dArea = np.loadtxt(input_roof_2dArea)
# roof_2dArea=roof_2dArea.astype(float)

input_roof_3dArea=r"E:\fyp on ccm computer\Maximum Likelihood Estimation\Maximum Likelihood Estimation\roof\3dArea.txt"
roof_3dArea = np.loadtxt(input_roof_3dArea)
roof_3dArea=roof_3dArea.astype(float)


input_roof_LDis=r"E:\fyp on ccm computer\Maximum Likelihood Estimation\Maximum Likelihood Estimation\roof\LDis.txt"
roof_LDis= np.loadtxt(input_roof_LDis)

input_roof_STDZ=r"E:\fyp on ccm computer\Maximum Likelihood Estimation\Maximum Likelihood Estimation\roof\std_Z.txt"
roof_STDZ= np.loadtxt(input_roof_STDZ)

#cal mean
# mean_roof_2dArea=np.mean(roof_2dArea)
mean_roof_3dArea=np.mean(roof_3dArea)
mean_roof_LDis=np.mean(roof_LDis)
mean_roof_STDZ=np.mean(roof_STDZ)

#cal variance
# var_roof_2dArea=np.var(roof_2dArea,ddof=1)
var_roof_3dArea=np.var(roof_3dArea,ddof=1)
var_roof_LDis=np.var(roof_LDis,ddof=1)
var_roof_STDZ=np.var(roof_STDZ,ddof=1)

#cal mean matrix
mean_matrix_roof=np.array([[mean_roof_3dArea],[mean_roof_LDis],[mean_roof_STDZ]])
# mean_matrix_roof=np.array([[mean_roof_2dArea],[mean_roof_3dArea],[mean_roof_LDis],[mean_roof_SDis],[mean_roof_STDX],[mean_roof_STDY],[mean_roof_STDZ]])
#cal cvc matrix
vcv_matrix_roof=np.cov(np.array([roof_3dArea,roof_LDis,roof_STDZ]))
# vcv_matrix_roof=np.cov(np.array([roof_2dArea,roof_3dArea,roof_LDis,roof_SDis,roof_STDX,roof_STDY,roof_STDZ]))

#cal probability x in a
transposeX_M_roof=(unknown_data-mean_matrix_roof).transpose() #[X-M]^T
inverse_vcv_roof=inv(vcv_matrix_roof)#V^-1
Times_roof=(transposeX_M_roof).dot(inverse_vcv_roof) #[X-M]^T*V^-1
Times_data_roof=Times_roof.dot(unknown_data-mean_matrix_roof)#[X-M]^T*V^-1*[X-M]
tempPX_roof=(1/2)*Times_data_roof#(1/2)*[X-M]^T*V^-1*[X-M]
det_vcv_roof=np.linalg.det(vcv_matrix_roof)#|V|

log_det_vcv_roof=math.log(det_vcv_roof)#loge|V|
log_times_e=(-1/2)*log_det_vcv_roof#(-1/2)*loge|V|
PX_roof=log_times_e-tempPX_roof#(-1/2)*loge|V|-(1/2)*[X-M]^T*V^-1*[X-M]
print(PX_roof)

#window
input_window_3dArea=r"E:\fyp on ccm computer\Maximum Likelihood Estimation\Maximum Likelihood Estimation\window\3dArea.txt"
window_3dArea = np.loadtxt(input_window_3dArea)
window_3dArea=window_3dArea.astype(float)

input_window_LDis=r"E:\fyp on ccm computer\Maximum Likelihood Estimation\Maximum Likelihood Estimation\window\LDis.txt"
window_LDis= np.loadtxt(input_window_LDis)

input_window_STDZ=r"E:\fyp on ccm computer\Maximum Likelihood Estimation\Maximum Likelihood Estimation\window\std_Z.txt"
window_STDZ= np.loadtxt(input_window_STDZ)

#cal mean
mean_window_3dArea=np.mean(window_3dArea)
mean_window_LDis=np.mean(window_LDis)
mean_window_STDZ=np.mean(window_STDZ)

#cal variance
var_window_3dArea=np.var(window_3dArea,ddof=1)
var_window_LDis=np.var(window_LDis,ddof=1)
var_window_STDZ=np.var(window_STDZ,ddof=1)

#cal mean matrix
mean_matrix_window=np.array([[mean_window_3dArea],[mean_window_LDis],[mean_window_STDZ]])
#cal cvc matrix
vcv_matrix_window=np.cov(np.array([window_3dArea,window_LDis,window_STDZ]))

#cal probability x in a
transposeX_M_window=(unknown_data-mean_matrix_window).transpose()
inverse_vcv_window=inv(vcv_matrix_window)
Times_window=(transposeX_M_window).dot(inverse_vcv_window)
Times_data_window=Times_window.dot(unknown_data-mean_matrix_window)
tempPX_window=(1/2)*Times_data_window
det_vcv_window=np.linalg.det(vcv_matrix_window)
log_det_vcv_window=math.log(det_vcv_window)#loge|V|
log_times_e=(-1/2)*log_det_vcv_window#(-1/2)*loge|V|
PX_window=log_times_e-tempPX_window#(-1/2)*loge|V|-(1/2)*[X-M]^T*V^-1*[X-M]
print(PX_window)

#Classification
classification=np.array(['Occlusion','Window','Rooftop'])
probability=np.array([PX_occlusion,PX_window,PX_roof])
index_max = np.argmax(probability)
classify=classification[index_max]
print('The hole detected is %s\n'%classify)