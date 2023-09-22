import cv2
import numpy as np
img=cv2.imread("C:\\Users\\ugcse.PG-CP.000\\Desktop\\210962018\\resource\\human.jfif", 0)
img=cv2.resize(img, (64, 128))
mag=np.zeros(img.shape, dtype='uint8')
ang=np.zeros(img.shape, dtype='uint8')
gx=img.copy()
gy=img.copy()
for i in range(1,img.shape[0]-1):
    for j in range(1,img.shape[1]-1):
        x=img[i,j+1]-img[i, j-1]
        y=img[i+1, j]-img[i-1, j]
        #print(x, y)
        mag[i, j]=int((x**2+y**2)**0.5)
        #print(y/x)
        #print(np.arctan(y/x))
        if x!=0 and y!=0:
            ang[i, j]=(np.arctan(y/x))
        else:
            ang[i, j]=0
        ang[i, j]=int(ang[i, j]*180/np.pi)
        gx[i, j]=x
        gy[i, j]=y
hist=np.zeros((16,8,9))
for i in range(16):
    for j in range(8):
        magblock=mag[i*8:(i+1)*8, j*8:(j+1)*8].copy()
        angblock=ang[i*8:(i+1)*8, j*8:(j+1)*8].copy()
        for x in range(0,8):
            for y in range(0,8):
                bin1=angblock[x,y]//20
                bin2=(bin1+1)%9
                weight=(angblock[x,y]%20)/20
                hist[i,j,bin1]+=magblock[x,y]*(1-weight)
                hist[i,j,bin2]+=magblock[x,y]*weight
print(hist)
features=np.zeros((15,7,36))
for i in range(15):
    for j in range(7):
        features[i,j,0:9]=hist[i,j].copy()
        features[i,j,9:18]=hist[i,j+1].copy()
        features[i,j,18:27]=hist[i+1,j].copy()
        features[i,j, 27:36]=hist[i+1,j+1].copy()
        k=((np.square(features[i,j])).sum())**0.5
        features[i,j]=features[i,j]/k
print(features)
cv2.imshow('Original', img)
cv2.imshow('Gx', gx)
cv2.imshow('Gy', gy)
cv2.imshow('Mag', mag)
cv2.imshow('Ang', ang)
cv2.waitKey(0)
cv2.destroyAllWindows()