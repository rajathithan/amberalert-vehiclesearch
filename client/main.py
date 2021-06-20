from kivy.app import App
from kivy.uix.camera import Camera
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout 
from kivy.uix.scatter import Scatter
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.widget import Widget
from kivy.uix.image import Image, AsyncImage
from kivy.clock import Clock
import socket
import os
from pathlib import Path
import threading
from threading import Thread
import time


start = 0
vehno = 0
vehtext = ""
setimage = 0 
addrstr = "some-address-value"

BUFF_SIZE = 7340032



class WindowManager(ScreenManager):
    def __init__(self, *args, **kwargs):
        super(WindowManager, self).__init__(*args, **kwargs)

class FirstWindow(Screen):
    def __init__(self, *args, **kwargs):
        super(FirstWindow, self).__init__(*args, **kwargs)
        
    def enableandsend(self, *args):
        self.ids.vehicleinput.hint_text = ''
        self.ids.startsearch.disabled = False
    
    def sendvehicleinfo(self, *args):
        global vehno, vehtext
        vehtext = self.ids.vehicleinput.text
        print(vehtext)
        vehno = 1

class SecondWindow(Screen):
    
    def __init__(self, *args, **kwargs):
        super(SecondWindow, self).__init__(*args, **kwargs)  
        
    def startclock(self, *args):
        Clock.schedule_interval(self.getTexture, 15)
      
        
    def getTexture(self, *args):       
        global start, configfilename, destfilename, orginalpath, jconfigfilename
        app = App.get_running_app()
        p = Path(app.user_data_dir)
        orginalpath = p.parent
        print(orginalpath)
        self.ids.camera.export_to_png(os.path.join(orginalpath, 'test.png'))
        configfilename = os.path.join(orginalpath, 'test.png')  
        start = 1


class ThirdWindow(Screen):
    
    def __init__(self, *args, **kwargs):
        super(ThirdWindow, self).__init__(*args, **kwargs)
    
    def startimageclock(self, *args):
        Clock.schedule_interval(self.setImage, 1)
        print("started-image-clock")
    

    def setImage(self, *args):
        global setimage, addrstr
        if setimage == 1:
            procimage = 'https://raj-5g-bucket.s3.amazonaws.com/' + addrstr + '.jpg'
            self.ids.cimage.source = procimage
            self.ids.cimage.reload()
            setimage = 0

class MySendingThread(Thread):
    def __init__(self, mySocket):
        Thread.__init__(self)
        self.mySocket = mySocket

    def run(self):
        global start, vehno, vehtext
        while True:
            if vehno!= 0:
                self.mySocket.send(bytes(vehtext, 'utf-8'))
                vehno = 0
                start = 1
                time.sleep(2)
            if start != 0:                
                try:
                    file = open(configfilename, 'rb')
                    image_data = file.read(BUFF_SIZE)        
                    self.mySocket.send(image_data)
                    file.close()
                    start = 0   
                    print(f"start - {start}")
                    print("sent")
                    #self.mySocket.send(bytes("sent", 'utf-8'))                 
                except:
                    pass
        



class MyReceivingThread(Thread):
    def __init__(self, mySocket):
        Thread.__init__(self)
        self.mySocket = mySocket

    def run(self):
        global start, setimage, addrstr
        while True:
            msg = self.mySocket.recv(1024)
            incomstr = msg.decode('utf-8')
            print(incomstr)
            incomstrlst = incomstr.split(',')
            if incomstrlst[0]  == 'disconnect':
                start = 0
                setimage = 1
                addrstr = incomstrlst[1]

            


#server_ip = '192.168.0.17'
server_ip = '155.146.96.72'

client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
client_socket.connect((server_ip, 9090))
client_socket.setblocking(0)

mySendThread = MySendingThread(client_socket)
myReceiveThread = MyReceivingThread(client_socket)

mySendThread.start()
myReceiveThread.start()
    
   

kv = Builder.load_file('main.kv')

class VehicleSearch(App):
    def build(self):
        
        return kv
    
 

# Start the Camera App

if __name__ == '__main__':
    VehicleSearch().run()  
      