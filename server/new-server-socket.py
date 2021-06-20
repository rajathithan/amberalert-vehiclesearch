import socket
import threading
from threading import Thread
import time
import os, sys
from  vehiclesearch import *
import logging

BUFF_SIZE = 7340032

didufind = 1
disconnect = 0

# create logger
logger = logging.getLogger('vehiclesearch in AWS Wavelength instance')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

class SendingThread(Thread):
    def __init__(self, mySocket, address):
        Thread.__init__(self)
        self.mySocket = mySocket
        self.address = address

    def run(self):
        global didufind, disconnect
        # write code to send data continuously
        while True:
            if didufind == 0 and disconnect == 0:
                adst = self.address.replace('.','-',3)
                sendstr = 'disconnect,'+ adst + ","
                print(sendstr)
                self.mySocket.send(bytes(sendstr,'utf-8'))
                logger.info('Sent disconnect notification to client - %s', self.address)
                disconnect = 1



class ReceivingThread(Thread):
    def __init__(self, mySocket, address):
        Thread.__init__(self)
        self.mySocket = mySocket
        self.address = address

    def run(self):
        global didufind, disconnect
        # write code to receive data continuously
        while True:
            try:
                msg = self.mySocket.recv(BUFF_SIZE)
                if msg:
                    if sys.getsizeof(msg) < 500:
                        print(sys.getsizeof(msg))
                        if not os.path.exists(self.address):
                            os.makedirs(self.address)
                        currentpath = os.getcwd()
                        vnewpath = os.path.join(currentpath, self.address, 'vehicle.txt')
                        vnewfile = open(vnewpath, "wt")
                        vehstr = msg.decode('utf-8')
                        vnewfile.write(vehstr)
                        logger.info('Received vehicle information - %s from the client - %s',vehstr, self.address)
                        time.sleep(1)
                        vnewfile.close()
                        msg = ""
                        didufind = 1
                    else:
                        if didufind != 0 :
                            if not os.path.exists(self.address):
                                os.makedirs(self.address)
                            currentpath = os.getcwd()
                            newpath = os.path.join(currentpath, self.address, 'image.png')
                            newfile = open(newpath, "wb")
                            newfile.write(msg)
                            newfile.close()
                            didufind = processimage(newpath, self.address, vehstr)
                            print(didufind)
                            if didufind == 0:
                                disconnect = 0
                            logger.info('Received image from the client - %s', self.address)
                            time.sleep(1)
                            msg = ""
            except:
                pass


server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


#server_ip = socket.gethostbyname(socket.gethostname())
server_ip = '192.168.0.17'
#server_ip = '10.0.1.91'
port = 9090
server_socket.bind((server_ip, port))

server_socket.listen()





def start():
    while True:
        mySocket, address = server_socket.accept()
        mySocket.setblocking(0)
        mySocket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
        if mySocket:
            receiveThread = ReceivingThread(mySocket, address[0])
            receiveThread.start()
            sendThread = SendingThread(mySocket, address[0])
            sendThread.start()


start()
