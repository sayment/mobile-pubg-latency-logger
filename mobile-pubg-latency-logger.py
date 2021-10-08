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


def main():
    video_ac()
    
    
    
def video_ac():
    #video urlsini kaydedin burada  \\ seklinde adresleme yapilmasına dikkat ediniz
    vid = cv2.VideoCapture("C:\\Users\\emre.baltaci\\Desktop\\python_isleri\\pubg2.mp4")
    if (vid.isOpened()== False):
        print("Video acilamadi, url yanlis olabilir kontrol ediniz")
    i=0
    kaydet = [["index","ms deger"]]
    while(vid.isOpened()):
        ret,frame = vid.read()#videoyu acar ret 0 donderse fail 1se saglikli
        if ret == True:
            #cv2.imshow('videos',frame)#framei videos basliginda ekrana basar
            resized = resize_video(frame)
            cv2.imshow('resized',resized)
            start_point = (92,540)
            end_point = (130,515)
            cropped = cv2.rectangle(resized, start_point, end_point, (255,0,0), 2)
            cropped_latency = latency_crop(frame,i)
            grayed_yaziyeri = make_clean(cropped_latency)
            text = save_latency_csv(grayed_yaziyeri)
            kaydet.append([i,text])
            i=i+1
            cv2.imshow('gray',grayed_yaziyeri)
            if cv2.waitKey(5) & 0xFF == ord('q'):#q inputu gelirse ekrandan çıkar
                break
        
        else:
            break
    
    np.savetxt("C:\\Users\\emre.baltaci\\Desktop\\python_isleri\\latency_listesi\\GFG.csv",kaydet,delimiter =", ",fmt ='% s')
    
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

def latency_crop(frame,counter):
    yaziyeri = frame[515*2:540*2,92*2:130*2]
    yazilacakyazi = 'C:\\Users\\emre.baltaci\\Desktop\\python_isleri\\latency_listesi\\' + 'cropped_latency_degeri_' + str(counter) + '.jpg'
    cv2.imwrite(yazilacakyazi,yaziyeri)
    return yaziyeri

def make_clean(yaziyeri):
    gray = cv2.cvtColor(yaziyeri, cv2.COLOR_BGR2GRAY)
    #filt = cv2.Canny(gray,100,150)
    #th, dst = cv2.threshold(gray, 50, 256, cv2.THRESH_BINARY);
    #return dst
    return gray

def save_latency_csv(gray):
    text = pytesseract.image_to_string(gray,lang="eng")
    return text
    
    
if __name__ == "__main__":
    main()
    
