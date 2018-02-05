#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os, sys
if sys.version_info[0] == 2:
    from Tkinter import * 
    from tkFont import Font
    from ttk import *
    #Usage:showinfo/warning/error,askquestion/okcancel/yesno/retrycancel
    from tkMessageBox import *
    #Usage:f=tkFileDialog.askopenfilename(initialdir='E:/Python')
    #import tkFileDialog
    #import tkSimpleDialog
else:  #Python 3.x
    from tkinter import *
    from tkinter.font import Font
    from tkinter.ttk import *
    from tkinter.messagebox import *
    #import tkinter.filedialog as tkFileDialog 
    #import tkinter.simpledialog as tkSimpleDialog    #askstring()
from PIL import ImageGrab,Image
import pickle
import neural_network 
import numpy as np
import matplotlib.pyplot as plt
import pylab 


#load the neural_network
pickle_file = open('saved_neuralnetwork.plk','rb')
savedNN = pickle.load(pickle_file)
pickle_file.close() 


class Application_ui(Frame):
    #这个类仅实现界面生成功能，具体事件处理代码在子类Application中。
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master.title('神经网络识别手写数字')
        self.master.geometry('445x416')
        self.createWidgets()

    def createWidgets(self):
        self.resulttext=StringVar()
        self.top = self.winfo_toplevel()

        self.style = Style()

        self.style.configure('Trec_button.TButton', font=('宋体',9))
        self.rec_button = Button(self.top, text='开始识别', command=self.rec_button_Cmd, style='Trec_button.TButton')
        self.rec_button.place(relx=0.593, rely=0.25, relwidth=0.164, relheight=0.079)

        self.num_canvas = Canvas(self.top, takefocus=1, bg='white')
       # self.num_canvas.place(relx=0.126, rely=0.25, relwidth=0.326, relheight=0.406)
        self.num_canvas.place(relx=0.126, rely=0.25, width=178, height=178)

        self.style.configure('Tcanvas_label.TLabel', anchor='w', font=('宋体',9))
        self.canvas_label = Label(self.top, text='请在此写下一个数字', style='Tcanvas_label.TLabel')
        self.canvas_label.place(relx=0.144, rely=0.192, relwidth=0.344, relheight=0.041)


        self.style.configure('TresultLabel.TLabel', anchor='w', font=('宋体',9))
        self.resultLabel = Label(self.top, text='神经网络识别的数字为：', style='TresultLabel.TLabel')
        self.resultLabel.place(relx=0.593, rely=0.404, relwidth=0.326, relheight=0.079)

        self.style.configure('Tbutton_redo.TButton', font=('宋体',9))
        self.button_redo = Button(self.top, text='重写', command=self.button_redo_Cmd, style='Tbutton_redo.TButton')
        self.button_redo.place(relx=0.216, rely=0.673, relwidth=0.164, relheight=0.079)

        self.style.configure('Ttitle.TLabel', anchor='center', font=('隶书',22))
        self.title = Label(self.top, text='手写数字识别系统', style='Ttitle.TLabel')
        self.title.place(relx=0.198, rely=0.038, relwidth=0.649, relheight=0.106)

        self.style.configure('Tresult.TLabel', anchor='w', font=('华文彩云',26))
        self.result = Label(self.top, textvariable=self.resulttext, style='Tresult.TLabel')
        self.result.place(relx=0.719, rely=0.519, relwidth=0.056, relheight=0.099)


class Application(Application_ui):
    #这个类实现具体的事件处理回调函数。界面生成代码在Application_ui中。
    def __init__(self, master=None):
        Application_ui.__init__(self, master)

    def rec_button_Cmd(self, event=None):
        img_data = []
        #TODO, Please finish the function here!
        #test_imag = self.grab_canvas_pic(self.num_canvas)
        image = self.grab_canvas_pic(self.num_canvas).convert("L").resize((28,28), Image.ANTIALIAS)
        #test_imag.save("./test.png")                                                                                                                                                                         
        img_list = list(image.getdata())
        img_data = 255.0 - np.asfarray(img_list)  #conventional for 0 to mean black and 255 to mean white, but MNIST data set has this the opposite way around

        #print(len(img_data))

        #plt.imshow(img_data.reshape((28,28)),cmap='Greys',interpolation='None')
        #pylab.show()

        img_data = img_data/255.0*0.99+0.01
        #print(img_data)

        outputs = savedNN.query(img_data)
        #the index of the highest value corresponds to the label
        label = np.argmax(outputs)
        print("the network answer is:", label)

        self.resulttext.set(label)

        pass

    def button_redo_Cmd(self, event=None):
        #TODO, Please finish the function here!
        self.num_canvas.delete("all")
        pass
    
    def paint(self,event):
        x1,y1 = (event.x-6),(event.y-6)
        x2,y2 = (event.x+6),(event.y+6)
        self.num_canvas.create_oval(x1,y1,x2,y2, fill = 'black',outline='black')
    
    def draw(self):
        self.num_canvas.bind("<B1-Motion>",self.paint)
    
    def grab_canvas_pic(self,widget):
        x = self.winfo_rootx() + widget.winfo_x()
        y = self.winfo_rooty() + widget.winfo_y()
        x1 = x + widget.winfo_width()
        y1 = y + widget.winfo_height()
        return ImageGrab.grab().crop((x,y,x1,y1))
        #return ImageGrab.grab().crop((x+2,y+2,x1-2,y1-2))


if __name__ == "__main__":
    top = Tk()
    Application(top).draw()

    mainloop()



