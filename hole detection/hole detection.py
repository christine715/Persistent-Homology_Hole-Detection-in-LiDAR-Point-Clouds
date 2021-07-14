import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
from mpl_toolkits.mplot3d import Axes3D
import math
import time
from pylab import figure
import pcl
import open3d as o3d

tic = time.time()

def poly_area2D(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
def caldistance(x1, y1, z1, x2, y2, z2):  
       d = math.sqrt(math.pow(x2 - x1, 2) +math.pow(y2 - y1, 2) +math.pow(z2 - z1, 2)* 1.0) 
       return d
def DrawHole(point,vertic_draw,position):
    fig = figure()
    ax = Axes3D(fig)
    for i in range(len(vertic_draw)):
        temp_vertic_draw=[]
        for item in vertic_draw:
            temp_vertic_draw.append(item)
            ax.scatter(point[item,0],point[item,1],point[item,2],color='b') 
            ax.text(point[item,0],point[item,1],point[item,2],  '%s' %item, size=20, zorder=1,color='k') 
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title(position)
    plt.show()
def cal_tri_area(hole_xyz):
    x1=hole_xyz[0,0]
    y1=hole_xyz[0,1]
    z1=hole_xyz[0,2]
    x2=hole_xyz[1,0]
    y2=hole_xyz[1,1]
    z2=hole_xyz[1,2]
    x3=hole_xyz[2,0]
    y3=hole_xyz[2,1]
    z3=hole_xyz[2,2]
    a = math.sqrt(math.pow(x2 - x1, 2) +math.pow(y2 - y1, 2) +math.pow(z2 - z1, 2)* 1.0) 
    b = math.sqrt(math.pow(x2 - x3, 2) +math.pow(y2 - y3, 2) +math.pow(z2 - z3, 2)* 1.0) 
    c = math.sqrt(math.pow(x3 - x1, 2) +math.pow(y3 - y1, 2) +math.pow(z3 - z1, 2)* 1.0) 
    s = (a+b+c)/2
    area = math.sqrt((s*(s-a)*(s-b)*(s-c)))
    return area
def poly_area3D(hole_xyz,LDistance):
    pcd = o3d.io.read_point_cloud(hole_xyz,format='xyz')
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)) 
    radius =LDistance
    radii = [radius/6,radius/3,radius, radius*3] 
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    triver=np.asarray(bpa_mesh.triangles)
    sum_area=0
    for count in range(len(triver)):
        tri=triver[count]
        hole_xyz=np.empty([3,3], dtype=float)
        hole_xyz[0,0]=point_cloud[tri[0],0]
        hole_xyz[0,1]=point_cloud[tri[0],1]
        hole_xyz[0,2]=point_cloud[tri[0],2] 
        hole_xyz[1,0]=point_cloud[tri[1],0] 
        hole_xyz[1,1]=point_cloud[tri[1],1] 
        hole_xyz[1,2]=point_cloud[tri[1],2]
        hole_xyz[2,0]=point_cloud[tri[2],0]
        hole_xyz[2,1]=point_cloud[tri[2],1]
        hole_xyz[2,2]=point_cloud[tri[2],2]
        sum_area=sum_area+cal_tri_area(hole_xyz)
    return sum_area


input_path=r"E:\fyp on ccm computer\result\1 point2\point\wen (suc represent by all hole)\word wen.txt"
point_cloud= np.loadtxt(input_path)
result = ripser(point_cloud,coeff=17, do_cocycles=True)
cocycles = result['cocycles']
hole=cocycles[1]
dgm=[]
diagrams = result['dgms']
dgms=diagrams[1]
dgm.append(dgms)

#1. Find lifetime for each hole
for dgm in dgm:
    l = dgm[:, 1] - dgm[:, 0]
    
index=sorted(range(len(l)), key=lambda k: l[k])
index=index[::-1]
#2. Order the intervals in B by decreasing length
l=np.sort(l)
l=l[::-1]
#3. Compute the persistent entropy HL of B
if all(l > 0):
    L = np.sum(l)
    p = l / L
    Hmax = -np.sum(p * np.log(p))#HL
#4. Compute the persistent entropy HL'(i)
#4.1 Compute l'k
#4.1.1 Compute e^(Hi)
Hi=[]
for a in range(len(l)):
    b=0
    temp_Hi=0
    while b<=a:
        temp_Hi=temp_Hi+((-l[b]/L)* np.log(l[b]/L))
        b=b+1
    Hi.append(temp_Hi)
EHi=[]    
for item in Hi:
    EHi.append(math.exp(item))
#4.1.2 Compute Si
lk=[] 
 
for a in range(len(l)):
    Sum=0
    b=0
    while b<=a:
        Sum=Sum+l[b]
        b=b+1
    lk.append(Sum/EHi[a])#l'k
    H2i=-(lk/L)* np.log(lk/L) #H'i
Hrel=[] #Hrel(i)
feature=[]
for a in range(len(l)):
    Hrel.append((H2i[a]-H2i[a-1])/(np.log(len(l))-Hmax))
    compare=((a-1)/len(l))
    if Hrel[a]>compare:
        feature.append(1)
    else :
        feature.append(0)  
feature=np.array(feature)
index=np.array(index)
Find_hole=np.empty((len(feature), 2))          
for a in range(len(feature)):
     Find_hole[a,0]=feature[a]
     Find_hole[a,1]=index[a] 
hole_feature=[]        
for a in range(len(Find_hole)):
    x_hole_feature=[]
    y_hole_feature=[]
    z_hole_feature=[]
    if Find_hole[a,0]==1:
        position=int(Find_hole[a,1])
        hole_feature.append(position)
        temp=hole[position]
        vertix_hole=[]
        for b in temp:
           for c in b:
             if all(vertix_hole!=c):
                vertix_hole.append(c)
        for d in range(len(vertix_hole)):
           vertix_index=vertix_hole[d]
           x_hole_feature.append(point_cloud[vertix_index,0])
           y_hole_feature.append(point_cloud[vertix_index,1])
           z_hole_feature.append(point_cloud[vertix_index,2])
        # DrawHole(point_cloud,vertix_hole,position) 
        distance=[]
        length_hole_feature=len(x_hole_feature)
        for h in range(length_hole_feature):
            for i in range(length_hole_feature-h):
                if h!=i+h:
                    distance.append(caldistance(x_hole_feature[h], y_hole_feature[h], z_hole_feature[h], x_hole_feature[i+h], y_hole_feature[i+h], z_hole_feature[i+h]))                 
        SDistance=min(distance)
        LDistance=max(distance)
        x_hole2=np.array(x_hole_feature)
        y_hole2=np.array(y_hole_feature)
        z_hole2=np.array(z_hole_feature)
        std_Z=np.std(z_hole2)
        #cal area
        hole_xyz=np.empty([len(x_hole2),3], dtype=float) 
        for j in range(len(x_hole2)):
            hole_xyz[j,0]=x_hole2[j]
            hole_xyz[j,1]=y_hole2[j]
            hole_xyz[j,2]=z_hole2[j]
        AreaOfHole3D=poly_area3D(hole_xyz,LDistance)
        AreaOfHole2D=poly_area2D(x_hole2,y_hole2)
        filename_coor='result_'+str(position)+'_'+str(a)+'.txt'
        with open(filename_coor,'w') as f:
            for e in range (len(x_hole_feature)):
                f.write('%s %s %s\n'%(x_hole_feature[e],y_hole_feature[e],z_hole_feature[e]))
        filename_distance='Distance_'+str(position)+'_'+str(a)+'.txt'
        with open(filename_distance,'w') as f:
            f.write('Shortest Distance= %s\n'%(SDistance))
            f.write('Longest Distance= %s\n'%(LDistance))
        filename_area='Area_'+str(position)+'_'+str(a)+'.txt'
        with open(filename_area,'w') as f:
            f.write('2D Area= %s\n'%(AreaOfHole2D))
            f.write('3D Area= %s\n'%(AreaOfHole3D))
        filename_stdz='STD_Z_'+str(position)+'_'+str(a)+'.txt'
        with open(filename_stdz,'w') as f:
            f.write('STD_Z= %s\n'%(std_Z))
   
plot_diagrams(diagrams, show = True)
plot_diagrams(diagrams, lifetime=True)     

#output permed point cloud
idx_perm=result['idx_perm']
fig = figure()
ax = Axes3D(fig)
ax.scatter(point_cloud[idx_perm,0],point_cloud[idx_perm,1],point_cloud[idx_perm,2])
plt.title("Subsampled Point Cloud")
plt.show()
# plt.subplot(222)
# idx_perm=result['idx_perm']
# plt.scatter(point_cloud[idx_perm,0],point_cloud[idx_perm,1])
# plt.title("Subsampled Cloud")
# plt.axis("equal")
# plt.show()
filename_perm='subsampled point cloud.txt'
with open(filename_perm,'w') as f:
    for e in idx_perm:
        f.write('%s %s %s\n'%(point_cloud[e,0],point_cloud[e,1],point_cloud[e,2]))


# # #output original point cloud
idx_perm=result['idx_perm']
fig = figure()
ax = Axes3D(fig)
ax.scatter(point_cloud[:,0],point_cloud[:,1],point_cloud[:,2])
plt.title("Original Point Cloud")
plt.show()

print ("success")
toc = time.time()
print(toc-tic)