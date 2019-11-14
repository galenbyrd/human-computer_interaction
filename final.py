from Leap import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import random
import globalVariables as gv
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib import gridspec

class Deliverable:
    def __init__ (self):
        self.controller = Controller()
        self.lines = []
        self.clf = pickle.load( open('userData/classifier.p','rb') )
        self.database = pickle.load(open('userData/database.p','rb'))
        self.testData = np.zeros((1,30),dtype='f')
        self.fig = plt.figure(figsize=(14,7))
        self.ax2 = self.fig.add_subplot(132,projection='3d')
        self.ax = self.fig.add_subplot(131)
        self.ax3 = self.fig.add_subplot(133)
        self.ax3.set_title('Rankings:')
        self.ax3.axis('off')
        self.ax3.set_ylim(0,50)
        self.ax2.set_xlim(-500,500)
        self.ax2.set_ylim(-400,400)
        self.ax2.set_zlim(0,600)
        self.ax2.view_init(azim=90)
        self.programState = 0
        self.currentNumberOfHands = 0
        self.sign = 0
        self.last=0
        self.lastRight=False
        self.imageBeenDisplayed=False
        self.timeExpired=False
        self.showProgress=False
        self.beenCentered=False
        self.attemptCounter=0
        self.lastAttemptCounter =0
        self.correctCounter=0
        self.meanx = -100
        self.meany = -100
        self.meanz = -100
        self.imageIndex = 10
        self.userName = 'No'
        self.userRecord = {'attempts'+str(self.sign):1}
        self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/leap.png')
        
    def CenterData(self):
        allx = self.testData[0,::3]
        ally = self.testData[0,1::3]
        allz = self.testData[0,2::3]
        self.meanx = allx.mean()
        self.meany = ally.mean()
        self.meanz = allz.mean()
        #print("x mean: ",self.meanx,"y mean: ",self.meany,"z mean: ",self.meanz)
        self.testData[0,::3] = allx - self.meanx
        self.testData[0,1::3] = ally - self.meany
        self.testData[0,2::3] = allz - self.meanz
        #print(self.testData[0,1::3])
        #print(self.testData[0][0],self.testData[0][27])

    def updateState(self):
        self.frame = self.controller.frame()
        self.currentNumberOfHands = len(self.frame.hands)
        previousState = self.programState
        if (self.currentNumberOfHands==0):
            self.programState = 0
        elif(self.currentNumberOfHands>0):
            self.programState = 1
            if (-75<self.meanx<75 and -75<self.meanz<75):
                self.programState = 2

    def drawHand(self):
        k = 0
        hand = self.frame.hands[0]
        for i in range(0,5):
            fingers = hand.fingers[i]
            for j in range (0,4):
                fingerBone = fingers.bone(j)
                positionOfBoneTip = fingerBone.next_joint
                positionOfBoneBase = fingerBone.prev_joint
                xBase = positionOfBoneBase[0]
                yBase = positionOfBoneBase[1]
                zBase = positionOfBoneBase[2]
                xTip = positionOfBoneTip[0]
                yTip = positionOfBoneTip[1]
                zTip = positionOfBoneTip[2]
                if(self.programState==1):
                    self.lines.append(self.ax2.plot([-xBase,-xTip],[zBase,zTip],[yBase,yTip],'r'))
                elif(self.lastRight==True):
                    self.lines.append(self.ax2.plot([-xBase,-xTip],[zBase,zTip],[yBase,yTip],'g'))
                else:
                    self.lines.append(self.ax2.plot([-xBase,-xTip],[zBase,zTip],[yBase,yTip],'y'))
                self.lastCounter=self.correctCounter
                if ( (j==0) | (j==3) ):
                    self.testData[0,k] = xTip
                    self.testData[0,k+1] = yTip
                    self.testData[0,k+2] = zTip
                    k = k + 3
        plt.pause(0.0001)
        while ( len(self.lines) > 0 ):
            ln = self.lines.pop()
            ln.pop(0).remove()
            del ln
            ln = []
 
    def HandleState0(self):
        self.correctCounter=0
        self.imageIndex=0
        self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/leap.png')
        image = plt.imread(self.image_file)
        self.ax.clear()
        self.ax.imshow(image)
        self.ax.set_title("")
        self.ax.axis('off')
        plt.pause(0.0001)
        self.updateState()

    def HandleState1(self):
        self.showProgress=False
        self.drawHand()
        self.CenterData()
        oldIndex=self.imageIndex
        if(self.meanx<-75):
            self.imageIndex=1
        elif(self.meanx>75):
            self.imageIndex=2
        elif(self.meanz<-75):
            self.imageIndex=3
        elif(self.meanz>75):
            self.imageIndex=4
        self.updateSubplot(oldIndex)
        self.updateState()

    def HandleState2(self):
        self.drawHand()
        self.CenterData()
        #print(self.testData)
        oldIndex=self.imageIndex
        self.imageIndex=5
        if (self.userRecord['attempts'+str(self.sign)]>=6):
            # SET TO MATH MODE
            self.imageIndex=8
        if (self.attemptCounter>50 and self.userRecord['attempts'+str(self.sign)]>=6):
            #TIME IS UP
            self.imageIndex=7
        if (self.testData[0][0]>self.testData[0][27]+40):
            #ROTATE HAND
            self.imageIndex=6
        if (self.meany<100):
            #MOVE HAND UP
            self.imageIndex=9
        if (self.testData[0][1]<-45):
            #FLATTEN HAND
            self.imageIndex=10
        if (self.timeExpired==True):
            self.imageIndex=5
        if(self.imageIndex==8 or self.imageIndex==5):
            predictedClass = self.clf.predict(self.testData)
            if (predictedClass == self.sign):
                self.correctCounter+=1
                self.lastRight=True
                if (self.showProgress==True):
                    for i in range(0, self.correctCounter):
                        self.ax3.axhspan(i, i+1, facecolor='g')
                    plt.show()
                if (self.correctCounter == 10):
                    self.HandleState3()
            else:
                self.lastRight=False
        self.updateSubplot(oldIndex)
        self.attemptCounter+=1
        self.updateState()

    def HandleState3(self):
        self.timeExpired=False
        self.lastAttemptCounter = self.attemptCounter
        self.attemptCounter=0
        self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/success.png')
        image = plt.imread(self.image_file)
        self.ax.clear()
        self.ax.imshow(image)
        self.ax.set_title("")
        self.ax.axis('off')
        plt.pause(1)
        self.correctCounter=0
        self.imageBeenDisplayed=False
        if (self.userRecord['attempts'+str(self.sign)]<7):
            if (self.sign<9):
                self.sign +=1
            else:
                self.sign=0
        else:
            self.sign = random.randint(0,9)
        try:
            self.userRecord['attempts'+str(self.sign)]+=1
        except:
            self.userRecord['attempts'+str(self.sign)]=1
        self.userRecord['lastNum']= self.sign
        pickle.dump(self.userRecord,open('userData/'+self.userName+'database.p','wb'))
        oldIndex=100
        self.updateSubplot(oldIndex)       

    def updateSubplot(self,oldIndex):
        if (self.imageIndex != oldIndex):
            self.ax.set_title("")
            self.ax.clear()
            if(self.imageIndex==1):
                #VISUALIZE MOVE RIGHT
                self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/right.png')
                image = plt.imread(self.image_file)
                self.ax.imshow(image)
                self.ax.axis('off')
                plt.pause(.7)
                self.ax.clear()
                self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/rightarrow.png')
                image = plt.imread(self.image_file)
                self.ax.imshow(image)
                self.ax.axis('off')
                plt.pause(.7)
                self.ax.clear()
                self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/straight.png')
                image = plt.imread(self.image_file)
                self.ax.imshow(image)
                self.ax.axis('off')
                plt.show()
            elif(self.imageIndex==2):
                #VISUALIZE MOVE LEFT
                self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/left.png')
                image = plt.imread(self.image_file)
                self.ax.imshow(image)
                self.ax.axis('off')
                plt.pause(.7)
                self.ax.clear()
                self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/leftarrow.png')
                image = plt.imread(self.image_file)
                self.ax.imshow(image)
                self.ax.axis('off')
                plt.pause(.7)
                self.ax.clear()
                self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/straight.png')
                image = plt.imread(self.image_file)
                self.ax.imshow(image)
                self.ax.axis('off')
                plt.show()
            elif(self.imageIndex==3):
                #VISUALIZE MOVE BACK
                self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/down.png')
                image = plt.imread(self.image_file)
                self.ax.imshow(image)
                self.ax.axis('off')
                plt.pause(.7)
                self.ax.clear()
                self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/downarrow.png')
                image = plt.imread(self.image_file)
                self.ax.imshow(image)
                self.ax.axis('off')
                plt.pause(.7)
                self.ax.clear()
                self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/straight.png')
                image = plt.imread(self.image_file)
                self.ax.imshow(image)
                self.ax.axis('off')
                plt.show()
            elif(self.imageIndex==4):
                #VISUALIZE MOVE FORWARD
                self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/up.png')
                image = plt.imread(self.image_file)
                self.ax.imshow(image)
                self.ax.axis('off')
                plt.pause(.7)
                self.ax.clear()
                self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/uparrow.png')
                image = plt.imread(self.image_file)
                self.ax.imshow(image)
                self.ax.axis('off')
                plt.pause(.7)
                self.ax.clear()
                self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/straight.png')
                image = plt.imread(self.image_file)
                self.ax.imshow(image)
                self.ax.axis('off')
                plt.show()
            elif(self.imageIndex==6):
                #VISUALIZE ROTATE HAND
                self.attemptCounter=0
                self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/rotateup.png')
                image = plt.imread(self.image_file)
                self.ax.imshow(image)
                self.ax.axis('off')
                plt.pause(.7)
                self.ax.clear()
                self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/rotatedown.png')
                image = plt.imread(self.image_file)
                self.ax.imshow(image)
                self.ax.axis('off')
                plt.pause(.7)
                self.ax.clear()
                self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/straight.png')
                image = plt.imread(self.image_file)
                self.ax.imshow(image)
                self.ax.axis('off')
                plt.show()
            elif(self.imageIndex==9):
                #VISUALIZE MOVE HAND UP
                self.attemptCounter=0
                self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/low.png')
                image = plt.imread(self.image_file)
                self.ax.imshow(image)
                self.ax.axis('off')
                plt.pause(.7)
                self.ax.clear()
                self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/uparrow.png')
                image = plt.imread(self.image_file)
                self.ax.imshow(image)
                self.ax.axis('off')
                plt.pause(.7)
                self.ax.clear()
                self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/straight.png')
                image = plt.imread(self.image_file)
                self.ax.imshow(image)
                self.ax.axis('off')
                plt.show()
            elif(self.imageIndex==10):
                #VISUALIZE FLATTEN HAND
                self.attemptCounter=0
                self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/handup.png')
                image = plt.imread(self.image_file)
                self.ax.imshow(image)
                self.ax.axis('off')
                plt.pause(.7)
                self.ax.clear()
                self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/straight.png')
                image = plt.imread(self.image_file)
                self.ax.imshow(image)
                self.ax.axis('off')
                plt.show()
            elif(self.imageIndex==8):
                # MATH
                # SHOW HAND
                plt.gcf().clear()
                self.ax2 = self.fig.add_subplot(132,projection='3d')
                self.ax2.set_xlim(-500,500)
                self.ax2.set_ylim(-400,400)
                self.ax2.set_zlim(0,600)
                self.ax2.view_init(azim=90)
                # SHOW EQUATION
                self.ax = self.fig.add_subplot(131)
                self.ax.set_xlim(0,5)
                self.ax.set_ylim(0,4)
                indicator = random.randint(0,1)
                if (indicator==0):
                    first = random.randint(1,15)
                    second = first - self.sign
                    display = str(first)+'-'+str(second)
                elif (indicator==1):
                    first = random.randint(1,15)
                    second = self.sign - first
                    display = str(second)+'+'+str(first)
                try:
                    display[4]
                    self.ax.text(-.2,1.5,display,fontsize=70)
                except:
                    self.ax.text(0,1.5,display,fontsize=100)
                #self.ax.set_title("Trial: "+str(self.userRecord['attempts'+str(self.sign)]))
                self.ax.axis('off')
                # SHOW RANKINGS
                self.ax3 = self.fig.add_subplot(133)
                self.rankings(self.ax3)
                plt.show() 
            elif(self.imageIndex==7):
                self.timeExpired=True
                #TIME'S UP!
                self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/clock.png')
                image = plt.imread(self.image_file)
                self.ax.imshow(image)
                self.ax.axis('off')
                plt.pause(.2)
                self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/clock2.png')
                image = plt.imread(self.image_file)
                self.ax.imshow(image)
                self.ax.axis('off')
                plt.pause(.2)
                self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/clock3.png')
                image = plt.imread(self.image_file)
                self.ax.imshow(image)
                self.ax.axis('off')
                plt.pause(.2)
                self.attemptCounter=0
#                newSign=random.randint(0,9)
#                while (self.sign==newSign):
#                  newSign=random.randint(0,9)
#                self.sign = newSign
            elif(self.imageIndex==5):
                if (self.beenCentered==False):
                    self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/success.png')
                    image = plt.imread(self.image_file)
                    self.ax.clear()
                    self.ax.imshow(image)
                    #self.ax.set_title("CENTERED FOR FIRST TIME")
                    self.ax.axis('off')
                    plt.pause(.7)
                    self.beenCentered=True
                if(self.userRecord['attempts'+str(self.sign)]<3):
                    self.showProgress=True
                    # DISPLAY HAND SIGN
                    plt.gcf().clear()
                    gs = gridspec.GridSpec(1, 4, width_ratios=[5,5,1,3])
                    # SHOW HAND
                    self.ax2 = plt.subplot(gs[1],projection='3d')
                    self.ax2.set_xlim(-500,500)
                    self.ax2.set_ylim(-400,400)
                    self.ax2.set_zlim(0,600)
                    self.ax2.view_init(azim=90)
                    # SHOW ASL DIGIT
                    self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/asl'+str(self.sign)+'.png')
                    image = plt.imread(self.image_file)
                    self.ax = plt.subplot(gs[0])
                    self.ax.imshow(image)
                    self.ax.set_title("Trial: "+str(self.userRecord['attempts'+str(self.sign)]))
                    self.ax.axis('off')
                    # SHOW PROGRESS
                    self.ax3 = plt.subplot(gs[2])
                    self.ax3.set_xlim(0,1)
                    self.ax3.set_ylim(0,10)
                    plt.xticks([])
                    plt.yticks([])
                    #SHOW RANKING
                    self.ax4 = plt.subplot(gs[3])
                    self.rankings(self.ax4)                   
                    plt.show()
                elif(self.timeExpired==False and self.userRecord['attempts'+str(self.sign)]>=4):
                    self.showProgress=False
                    # DISPLAY ONLY DIGIT
                    # SHOW HAND
                    plt.gcf().clear()
                    self.ax2 = self.fig.add_subplot(132,projection='3d')
                    self.ax2.set_xlim(-500,500)
                    self.ax2.set_ylim(-400,400)
                    self.ax2.set_zlim(0,600)
                    self.ax2.view_init(azim=90)
                    # SHOW DIGIT
                    self.ax = self.fig.add_subplot(131)
                    self.ax.set_xlim(0,5)
                    self.ax.set_ylim(0,4)
                    self.ax.text(1.5,1.5,str(self.sign),fontsize=150)
                    self.ax.set_title("Trial: "+str(self.userRecord['attempts'+str(self.sign)]))
                    self.ax.axis('off')
                    #SHOW RANKING
                    self.ax3 = self.fig.add_subplot(133)
                    self.rankings(self.ax3)
                    plt.show()
                else:
                    # DISPLAY SIGN FOR 2 SECONDS
                    self.showProgress=True
                    plt.gcf().clear()
                    gs = gridspec.GridSpec(1, 4, width_ratios=[5,5,1,3])
                    # SHOW HAND
                    self.ax2 = plt.subplot(gs[1],projection='3d')
                    self.ax2.set_xlim(-500,500)
                    self.ax2.set_ylim(-400,400)
                    self.ax2.set_zlim(0,600)
                    self.ax2.view_init(azim=90)
                    # SHOW RANKINGS
                    self.ax4 = plt.subplot(gs[3])
                    self.rankings(self.ax4) 
                    # SHOW DIGIT
                    self.ax = plt.subplot(gs[0])
                    if(self.imageBeenDisplayed==False):
                        self.imageBeenDisplayed=True
                        self.image_file = cbook.get_sample_data('/Users/galenbyrd/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib/images/asl'+str(self.sign)+'.png')
                        image = plt.imread(self.image_file)
                        self.ax.imshow(image)
                        self.ax.set_title("Trial: "+str(self.userRecord['attempts'+str(self.sign)]))
                        self.ax.axis('off')
                        plt.pause(2)
                        self.ax.clear()
                    self.ax.set_xlim(0,5)
                    self.ax.set_ylim(0,4)
                    self.ax.text(1.5,1.5,str(self.sign),fontsize=150)
                    self.ax.set_title("Trial: "+str(self.userRecord['attempts'+str(self.sign)]))
                    self.ax.axis('off')
                    # SHOW PROGRESS
                    self.ax3 = plt.subplot(gs[2])
                    self.ax3.set_xlim(0,1)
                    self.ax3.set_ylim(0,10)
                    plt.xticks([])
                    plt.yticks([])
                    plt.show()
                               
    def rankings(self,subFig):
        temp={}
        i=48
        #print('Rankings:')
        subFig.set_title('Rankings:')
        subFig.axis('off')
        subFig.set_ylim(0,50)
        for key in self.database.keys():
            try:
                currentUserRecord = pickle.load(open('userData/'+key+'database.p','rb'))
                rank=sum(currentUserRecord.values())-currentUserRecord['lastNum']-1
                temp[key]=rank
            except:
                temp[key]=0
        for key,value in sorted(temp.iteritems(), key=lambda (k,v): (v,k),reverse=True):
            if (key==self.userName):
                #print ("%s: %s         last time: %s" % (key, value, self.last))
                subFig.text(0,i,"%s: %s           previously: %s" % (key, value, self.last))
            else:
                #print "%s: %s" % (key, value)
                subFig.text(0,i,"%s: %s" % (key, value))
            i-=2

    def runOnce(self):
        #print(self.testData)
        self.updateState()
        if (self.programState==0):
            self.HandleState0()
        elif(self.programState==1):
            self.HandleState1()
        elif(self.programState==2):
            self.HandleState2()

    def runForever(self):
        #LOGIN
        self.userName = raw_input('Please enter your name: ')
        if self.userName in self.database:
            self.database[self.userName]['logins']+=1
            self.userRecord = pickle.load(open('userData/'+self.userName+'database.p','rb'))
            self.sign = self.userRecord['lastNum']
            self.last=sum(self.userRecord.values())-self.userRecord['lastNum']-1
            print 'welcome back ' + self.userName
        else:
            self.database[self.userName] = {'logins':1}
            print 'welcome ' + self.userName
            self.userRecord['lastNum']= 0
            pickle.dump(self.userRecord,open('userData/'+self.userName+'database.p','wb'))
        #print(self.database)
        #print(self.userRecord)
        pickle.dump(self.database,open('userData/database.p','wb'))

        matplotlib.interactive(True)
        image = plt.imread(self.image_file)
        self.ax.imshow(image)
        self.ax.axis('off')
        self.rankings(self.ax3)
        plt.pause(0.00001)
        while (True):
            self.runOnce()
                        
deliverable=Deliverable()
deliverable.runForever()       

