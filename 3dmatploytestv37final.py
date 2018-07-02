import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D




fig = plt.figure(figsize=(16,10))

#fig = plt.figure()




#ax = fig.add_subplot(311, projection='3d')

#fig.set_size_inches (30,20)


img = mpimg.imread("/home/pi/Desktop/testimage1.jpg")
               


B = img[:,:,2]
G = img[:,:,1]
R = img[:,:,0]

y1=1175
y2=1487

x1=1045
x2=1547


#ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.0, 1.0, 1, 1]))


#pixels =  B[00:1600, 610]
pixels = B[y1:y2, x1:x2]
#print len (pixels)

pixelsG = G[880:920, x1:x2]



Y = np.arange (y1,y2)

#X= np.arange (1,len(pixels[0])+1)

X= np.arange (x1,x2)
#for x in range(10):
    #X.append(np.arange (1,len(pixels[0])+1))
    #Y.append(np.arange (1,len(pixels[0])+1))    

X,Y = np.meshgrid(X,Y) 
print (Y)

print (pixels[0], len(pixels[0]))

XX,YY = np.meshgrid(X, Y, sparse=True)
Z = pixels


#figure mesh(X,Y,Z,C,'FaceLighting','gouraud','LineWidth',0.3)


# Plot the surface.1

fig.subplots_adjust(hspace=0.3)
ax1 = fig.add_subplot(3,1,1, projection='3d')
ax1.set_zlabel('Blue')
ax1.tick_params(axis='x',colors="blue")


surf1 = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
ay1= plt.ylim (y2,y1)
ax1.tick_params(axis='y',colors="blue")
#ax1.set_zticks([0,1,2,4,8,16,32,64,128,256])
#fig.colorbar(1)

# Plot the surface 2
ax2 = fig.add_subplot(3,1,2, projection='3d')
Z = G[y1:y2, x1:x2]
ay2= plt.ylim (y2,y1)

ax2.set_zlabel('Green')


ax2.tick_params(axis='x',colors="green")
ax2.tick_params(axis='y',colors="green")

surf2 = ax2.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)

# Plot the surface 3

Z = R[y1:y2, x1:x2]
ax3 = fig.add_subplot(3,1,3,projection='3d')


ay3= plt.ylim (y2,y1)

ax3.set_zlabel('Red')
ax3.set_xlabel('Horizontal - portion displayed out of 3280 ')
ax3.set_ylabel('Vertical -portion  displayed out of 2464')

ax3.tick_params(axis='x',colors="red")
ax3.tick_params(axis='y',colors="red")

ax3.xaxis.labelpad =30
ax3.yaxis.labelpad =20

surf3 = ax3.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)







plt.tight_layout()







#plt.setp([a.get_xticklabels() for a in ax[0, :]], visible=False)
#plt.setp([a.get_yticklabels() for a in ax[:, 1]], visible=False)



#plt.savefig("sample.png",bbox_inches='tight',dpi=100)

#plt.show()


#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

#fig.colorbar(surf1, shrink=0.5, aspect=5)


#h = plt.contourf(X,Y,Z)









plt.savefig("/home/pi/Desktop/3dtestsmall section.png")















f, axarr = plt.subplots(2, 2)
axarr[0,0].imshow(img, cmap = cm.Greys_r)
axarr[0,0].set_title("RGB")
axarr[0,0].axis('on')

axarr[0,1].imshow(B, cmap = cm.Greys_r)
axarr[0,1].set_title("Blue")
axarr[0,1].axis('on')

axarr[1,0].imshow(G, cmap = cm.Greys_r)
axarr[1,0].set_title("Green")
axarr[1,0].axis('on')

axarr[1,1].imshow(R, cmap = cm.Greys_r)
axarr[1,1].set_title("Red")
axarr[1,1].axis('on')


# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)


plt.tight_layout()

plt.show()



