# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 09:36:04 2021

@author: emre.baltaci
"""

import cv2
import numpy as np
import pandas as pd
import pytesseract
import csv
import matplotlib.pyplot as plt
import os

def main():
    
    isim = "Huawei OptiXstar K562-Yok-Mobil-Var-5 GHz"
    
    konum = "C:\\Users\\emre.baltaci\\Desktop\\python_isleri\\Huawei gaming 5ghz\\"
    
    log = open(konum+isim+".txt", mode="w")
    
    video_ac(log,isim,konum)

def nothing(x):
    pass
    
def video_ac(log,isim,konum):
    #video urlsini kaydedin burada  \\ seklinde adresleme yapilmasına dikkat ediniz
    
    
    
    adres = konum + isim + ".mp4"
    vid = cv2.VideoCapture(adres)
    if (vid.isOpened()== False):
        print("Video acilamadi, url yanlis olabilir kontrol ediniz")
    i=0
    kaydet = [["index","ms deger"]]
    #cv2.namedWindow('gray')

# create trackbars for color change
    #cv2.createTrackbar('R1','gray',0,255,nothing)
    #cv2.createTrackbar('G1','gray',0,255,nothing)
    #cv2.createTrackbar('B1','gray',0,255,nothing)
    #cv2.createTrackbar('R2','gray',0,255,nothing)
    #cv2.createTrackbar('G2','gray',0,255,nothing)
    #cv2.createTrackbar('B2','gray',0,255,nothing)
    #switch = '0 : OFF \n1 : ON'
    #cv2.createTrackbar(switch, 'gray',0,1,nothing)

    while(vid.isOpened()):
        ret,frame = vid.read()#videoyu acar ret 0 donderse fail 1se saglikli
        if ret == True:
            #cv2.imshow('videos',frame)#framei videos basliginda ekrana basar
            resized = resize_video(frame)
            #cv2.imshow('resized',resized)
            #start_point = (92,540)
            #end_point = (130,515)
            #cropped = cv2.rectangle(resized, start_point, end_point, (255,0,0), 2)
            cropped_latency = latency_crop(frame)
            #cv2.imshow('cropped',cropped_latency)
            grayed_yaziyeri = make_clean(cropped_latency,i,konum,isim)
            text = save_latency_csv(grayed_yaziyeri)
            if text.find('ms') != -1:
                x=text.split("ms")
                #log.writelines(str(i) + ", " + x[0] + "\n")
                try:
                    log.writelines(str(i) + ", " +str(int(x[0]))[:2] + "\n")
                    kaydet.append([i,str(int(x[0]))[:2]])
                    
                except Exception:
                    pass
                
            
            
            i=i+1
            cv2.imshow('gray',grayed_yaziyeri)
            if cv2.waitKey(5) & 0xFF == ord('q'):#q inputu gelirse ekrandan çıkar
                break
        
        else:
            break
    
    log.close()
    konumcsv = konum + isim + ".csv"
    np.savetxt(konumcsv,kaydet,delimiter =", ",fmt ='% s')
    
    
    #plt.plot(kaydet[1:len(kaydet)][0],kaydet[1:len(kaydet)][1])
    #plt.show()

    #kaydet.to_csv('C:\\Users\\emre.baltaci\\Desktop\\python_isleri\\latency_listesi\\ms_listesi.csv')
    vid.release()
    cv2.destroyAllWindows()
    
    
    
def resize_video(frame):
    olcu = 50
    genislik = int(frame.shape[1]*olcu/100)
    yukseklik = int(frame.shape[0]*olcu/100)
    dim = (genislik, yukseklik)
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return resized

def latency_crop(frame):
    yaziyeri = frame[300:335,2200:2280]
    olcu = 300
    genislik = int(yaziyeri.shape[1]*olcu/100)
    yukseklik = int(yaziyeri.shape[0]*olcu/100)
    dim = (genislik, yukseklik)
    yaziyeri = cv2.resize(yaziyeri, dim, interpolation = cv2.INTER_AREA)
    return yaziyeri

def make_clean(yaziyeri,counter,konum,isim):
    hsv = cv2.cvtColor(yaziyeri, cv2.COLOR_BGR2HSV)
    lower = np.array([40,130,31], dtype = "uint8")
    upper = np.array([75,255,255], dtype = "uint8")
    #lower = np.array([cv2.getTrackbarPos('B1','gray'),cv2.getTrackbarPos('G1','gray'),cv2.getTrackbarPos('R1','gray')], dtype = "uint8")
    #upper = np.array([cv2.getTrackbarPos('B2','gray'),cv2.getTrackbarPos('G2','gray'),cv2.getTrackbarPos('R2','gray')], dtype = "uint8")
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(yaziyeri, yaziyeri, mask = mask)
    dst = cv2.GaussianBlur(output,(3,3),cv2.BORDER_DEFAULT)
    #ret, thresh1 = cv2.threshold(dst, 130, 255, cv2.THRESH_BINARY_INV)
    konumscreenshot = konum + "latency_listesi_" + isim + "\\"
    try:
        os.mkdir(konumscreenshot)
    except OSError:
        pass
    yazilacakyazi = konumscreenshot + "cropped_latency_degeri_" + str(counter) + ".jpg"
    cv2.imwrite(yazilacakyazi,dst)
    #filt = cv2.Canny(gray,100,150)
    #th, dst = cv2.threshold(gray, 50, 256, cv2.THRESH_BINARY);
    #return dst
    return dst

def save_latency_csv(gray):
    text = pytesseract.image_to_string(gray,lang="eng")
    return text
    
    
if __name__ == "__main__":
    main()
    
