# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 09:36:04 2021

@author: emre.baltaci
"""

import cv2
import numpy as np



def main():
    video_ac()
    
    
    
def video_ac():
    #video urlsini kaydedin burada  \\ seklinde adresleme yapilmasına dikkat ediniz
    vid = cv2.VideoCapture("C:\\Users\\emre.baltaci\\Desktop\\python_isleri\\pubgdeneme.mp4")
    if (vid.isOpened()== False):
        print("Video acilamadi, url yanlis olabilir kontrol ediniz")
    i=1
    for i in range(1):#while(vid.isOpened())
        ret,frame = vid.read()#videoyu acar ret 0 donderse fail 1se saglikli
        if ret == True:
            cv2.imshow('videos',frame)#framei videos basliginda ekrana basar
            resized = resize_video(frame)
            cv2.imshow('resized',resized)
            start_point = (92,540)
            end_point = (130,515)
            cropped = cv2.rectangle(resized, start_point, end_point, (255,0,0), 2)
            
            cropped_latency = latency_crop(frame,i)
            
            if cv2.waitKey(2000) & 0xFF == ord('q'):#q inputu gelirse ekrandan çıkar
                break
        
        else:
            break
    
    vid.release()
    cv2.destroyAllWindows()

def resize_video(frame):
    olcu = 50
    genislik = int(frame.shape[1]*olcu/100)
    yukseklik = int(frame.shape[0]*olcu/100)
    dim = (genislik, yukseklik)
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return resized

def latency_crop(frame,counter):
    yaziyeri = frame[515*2:540*2,92*2:130*2]
    yazilacakyazi = 'C:\\Users\\emre.baltaci\\Desktop\\python_isleri\\latency_listesi\\' + 'cropped_latency_degeri_' + str(counter) + '.jpg'
    cv2.imwrite(yazilacakyazi,yaziyeri)
    return yaziyeri
if __name__ == "__main__":
    main()
    
