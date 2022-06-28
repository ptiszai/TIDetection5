# by Tiszai Istvan, GPL-3.0 license

from queue import Queue, Empty
from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from PIL import ImageTk, Image
from tkinter.messagebox import showinfo
import os
from os import walk
import cv2
import torch
from tidetect import (Detect)
from pprint import pprint
from utils.general import YolovX

class MyException(Exception):
    pass
# raise DriverException('Driver not found.')

class mainGUI(object):
    def __init__(self, master, w_w = 1024, W_h = 768):
        # construct
        # main window
        self.master = master
        window_width = w_w
        window_height = W_h
        self.yolovX = YolovX.none
        #self.yolovX_prev = YolovX.none
        self.yolovX_model=""
        #self.yolovX_model_prev=""
        self.filename_image = ""
        self.model=None
        self.dt=None
        self.nosave_image=True
        self.set_detectArgumentums_default()       
        #self.CLASSES = None
        #self.current_modelName = "ssd300_vgg16"
        master.title("Detection GUI")
        #self.fileStream = ""
        (c_x, c_y) = canter(master, window_width, window_height)
        master.geometry(f'{window_width}x{window_height}+{c_x}+{c_y}')
        master.resizable(False, False)
        # menu
        self.menubar = Menu(self.master)
        self.menus()
        self.master.config(menu=self.menubar)
        # configure the grid
        self.master.columnconfigure(0, weight=1)
        self.master.columnconfigure(1, weight=3)      
        #self.queue = Queue()       
        #self.fileStream=fvsThr.FileVideoStream(self, self.queue)
        #self.algorithm=algoThr.TIalgo(self, self.queue)
        self.create_widgets()   
        #self.algorithm.thread.start()
        return

    def menus(self):          
        self.filemenu = Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="Open", command=self.select_file)       
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=self.master.quit)
        self.menubar.add_cascade(label="File", menu=self.filemenu)
        # About
        self.info = Menu(self.menubar, tearoff=0)
        self.info.add_command(label="About", command=self.about) 
        self.menubar.add_cascade(label="Info", menu=self.info)        

    def create_widgets(self):
        _frameLeft = Frame(self.master,bg="aliceblue").place(relx=0, rely=0, relwidth=0.25, relheight=0.95)    
         #self.frameLeft.place(relx=0, rely=0, relwidth=0.25, relheight=0.95)        
        frameRight = Frame(self.master,bg="gray")
        frameRight.place(relx=0.25, rely=0, relwidth=0.75, relheight=0.75)  
        frameBottomRight = Frame(self.master,bg="aquamarine2")
        frameBottomRight.place(relx=0.25, rely=0.75, relwidth=0.75, relheight=0.20) 
        frameBottom = Frame(self.master,bg="beige")
        frameBottom.place(relx=0, rely=0.95, relwidth=1, relheight=0.05)
        # bottom labels
        self.stateLabel1 = Label(frameBottom, text="---")
        self.stateLabel1.place(in_=frameBottom, x=10, y=1)
        #self.stateLabel2 = Label(frameBottom, text="---")
        #self.stateLabel2.place(in_=frameBottom, x=10, y=17)
        # video, camera start, stop buttons
        self.stop_btn =  Button(frameBottomRight, text="Stop", relief=RIDGE, state=DISABLED, font={"Arial", 12}, command=self.stop_video)
        self.stop_btn.pack(side=RIGHT, anchor=tk.NE, padx=10, pady=10)
        self.start_btn =  Button(frameBottomRight, text="Start", relief=RIDGE, state=DISABLED, font={"Arial", 12}, command=self.start_video)
        self.start_btn.pack(side=RIGHT, anchor=tk.NE, padx=10, pady=10)
        #self.start_btn.configure(state=DISABLED)  
        #self.stop_btn.configure(state=DISABLED)
        # image label
        self.imageVideo_label = Label(frameRight)
        self.imageVideo_label.place(in_=frameRight, anchor="c", relx=.5, rely=.5)
         #self.insert_image_to_frame( "../imagesVideos/640x480-azure-mist-solid-color-background.jpg", True)
        # Checkbutton
        self.var_cbt = tk.IntVar()
        self.var_cbt.set(-1)
        self.camera_checkbtn = Checkbutton(frameBottomRight,text='Camera',command=self.camera_enable,variable=self.var_cbt,onvalue=0,offvalue=-1, bg="aquamarine2")
        self.camera_checkbtn.pack(side=LEFT, anchor=tk.NW, padx=10, pady=10)
        self.var_saving_cbt = tk.IntVar()
        self.var_saving_cbt.set(-1)
        self.save_checkbtn = Checkbutton(frameBottomRight,text='Saving images',command=self.saving_enable,variable=self.var_saving_cbt,onvalue=0,offvalue=-1, bg="aquamarine2")
        self.save_checkbtn.pack(side=LEFT, anchor=tk.NW, padx=10, pady=10)
        self.save_checkbtn.configure(state=DISABLED)
       
        self.var_cbt.set(-1)
        #self.refresh_btn = Button(frameBottomRight, text="Refresh", relief=RIDGE, state=DISABLED, font={"Arial", 12}, command=self.refresh_image)
        #self.refresh_btn.pack(side=RIGHT, anchor=tk.NW, padx=10, pady=10)         
        #start_btn.place(in_=frameBottomRight, x=10, y=1)
        #stop_btn = Button(frameBottomRight, "Stop", command=self.donothing)
        _title_label = Label(_frameLeft, text="Object Detections YOLOv5 with Deep MI\n Deep-learning Machine Intelligence Network", fg="red")
        _title_label.place(in_=_frameLeft, x=1, y=5)
        #_title_label.pack(side=TOP, anchor=tk.NW, padx=1, pady=10)
        #_title_label = Label(_frameLeft, text="Object Detections YOLOvX with Deep MI")
        _title_label_1 = Label(_frameLeft, text="Real-Time Object Detection Algorithm:")
        _title_label_1.place(in_=_frameLeft, x=1, y=40) 
        
        # Radiobutton
        self.var_rdb = IntVar()
        self.none_rdb = Radiobutton(_frameLeft, text="none", variable=self.var_rdb, value=0, command=self.yolovx_rb_change)
        self.none_rdb.place(in_=_frameLeft, x=10, y=60)
        self.y1_rdb = Radiobutton(_frameLeft, text="yolov5", variable=self.var_rdb, value=1, command=self.yolovx_rb_change)
        self.y1_rdb.place(in_=_frameLeft, x=10, y=80)

        _title_label_2 = Label(_frameLeft, text="Models:")
        _title_label_2.place(in_=_frameLeft, x=1, y=120)   
        # model Combobox
        self.var_cbx = StringVar()        
        self.yx_cbx = ttk.Combobox(_frameLeft, textvariable=self.var_cbx, width=35)
        # get first 3 letters of every month name
        #self._yx_cbx['values'] = [month_name[m][0:3] for m in range(1, 13)]
        # prevent typing a value
        self.yx_cbx['state'] = 'readonly'
        # place the widget
        self.yx_cbx.place(in_=_frameLeft, x=10, y=140)
        #_yx_cbx.pack(side=LEFT, anchor=tk.N, padx=10, pady=30)
        self.yx_cbx.bind('<<ComboboxSelected>>', self.yx_cbx_changed)
        self.yx_cbx.configure(state=DISABLED)
        self.setModel(yolovx=YolovX.none)       
        #_, _, filenames = next(walk("./weights"), (None, None, []))
        self.yx_cbx['values'] = []
        # load model
        self.model_load_btn =  Button(_frameLeft, text="Model load", relief=RIDGE, state=DISABLED, command=self.model_load)
        self.model_load_btn.place(in_=_frameLeft, x=10, y=163)
        self.model_load_btn.configure(state=DISABLED)

        # left text box
        _text_frame = Frame(_frameLeft,  height=2,width=29)
        _text_frame.place(in_=_frameLeft, x=10, y=210)
        self.scroll_text_box = Scrollbar(_text_frame,orient='horizontal')
        self.scroll_text_box.pack(side=BOTTOM, fill=X)
        #scroll_text_box.place(in_=_frameLeft, x=10, y=210,  height=2,width=29)
        self.text_box = Text(_text_frame, height=31,width=29,relief=RAISED, wrap = "none",xscrollcommand=self.scroll_text_box.set)
        #self.text_box.place(in_=_text_frame, x=10, y=200)
        self.text_box.pack()
        #self.text_box.config(state='disabled')
        #self.text_box.config(xscrollcommand=scroll_text_box.set)
        self.scroll_text_box.config(command=self.text_box.xview)
        #self.text_box.insert('end', "123456789asdfghjklpoiuytr123456789o")

        # youtube part
        _youtube_label = Label(frameBottomRight, text="Youtube link:CTRL-C+CTRL-V -> 'Set' press -> 'Start' press'", bg="aquamarine2")
        _youtube_label.place(in_=frameBottomRight, x=10, y=35)
        self.youtube_text = Text(frameBottomRight, height=1,width=60,relief=SUNKEN, wrap = "none")
        self.youtube_text.place(in_=frameBottomRight, x=10, y=55)
        self._youtube_set_btn =  Button(frameBottomRight, text="Set", relief=RIDGE, command=self.youtube_set)
        self._youtube_set_btn.place(in_=_frameLeft, x=500, y=52)
        self._youtube_set_btn.configure(state=DISABLED)
        return

    def select_file(self):
        if not torch.cuda.is_available() :
             showinfo(title='CUDA',message="Not founded CUPA CPU")
             return
        _filetypes = (('Jpg files', '*.jpg'),('Png files', '*.png'),('Mp4 files', '*.mp4'),('All files', '*.*'))
        _filename = fd.askopenfilename(title='Open a file',initialdir='../imagesVideos',filetypes=_filetypes)
        if not _filename:
            showinfo(title='Selected File',message="filename empty")
            return        
        _f_ext = os.path.splitext(_filename.lower())
        self.clear_youtube_text()
        if _f_ext[1] == ".mp4":           
            self.filename_image = _filename
            #self.refresh_btn.configure(state=DISABLED)
            self.start_stop(start=NORMAL, stop=NORMAL)
            #self.start_btn.configure(state=NORMAL)  
            #self.stop_btn.configure(state=DISABLED)
        else:
            self.filename_image = _filename
            self.set_detectArgumentums_default()
            self.detectArgumentums['source']=_filename
            self.detectArgumentums['nosave']=self.nosave_image
            #self.detectArgumentums['weights']='weights/COCO-Detection/yolov5n6.pt'
            #self.detectArgumentums['model']=self.model
            self.dt=Detect(self, self.detectArgumentums)
            self.dt.Start(modul=self.model, source=self.filename_image)

            #detect(self.detectArgumentums, self)
            #self.insert_image_to_frame(_filename)
            #self.refresh_btn.configure(state=NORMAL)
            self.start_stop(start=DISABLED, stop=NORMAL)
            #self.start_btn.configure(state=DISABLED)  
            #self.stop_btn.configure(state=DISABLED)

    def quit(self):
        self.master.destroy()                   

    def image_to_frame(self, image):     
        if not image is None:
            (_b,_g,_r) = cv2.split(image)        
            image = cv2.merge((_r,_g,_b))
            # Convert the Image object into a TkPhoto object
            _image = Image.fromarray(image)
            _photo_img = ImageTk.PhotoImage(image=_image)
            self.imageVideo_label.configure(image=_photo_img)
            self.imageVideo_label.image=_photo_img     
            self.master.update_idletasks()
    
    def start_video(self):
        self.start_stop(start=DISABLED, stop=NORMAL)
        #self.start_btn.configure(state=DISABLED)  
        #self.stop_btn.configure(state=NORMAL)
        self.camera_checkbtn.configure(state=DISABLED)
        self.algoButtonsEnabled(enabled=False)
        #_modelName = None
        #if self.yolovX_model != "" and self.yolovX_model != self.yolovX_model_prev :
        #    _modelName = self.yolovX_model
        #self.algorithm.start(filename=self.fullfilename, camera=self.var_cbt.get(), modelname = _modelName)
        #self.fileStream.start(filename=self.fullfilename, camera=self.var_cbt.get())
        self.detectArgumentums['source'] = '0' if self.var_cbt.get() > -1 else self.filename_image
        #self.detectArgumentums['source']=self.filename_image          
        self.detectArgumentums['model']=self.model
        self.detectArgumentums['nosave']=self.nosave_image
        self.dt=Detect(self, self.detectArgumentums)
        self.dt.Start(modul=self.model, source=self.detectArgumentums['source'])
        #self.dt.update()
        
    def stop_video(self):   
        self.camera_checkbtn.configure(state=NORMAL)
        self.algoButtonsEnabled(enabled=True)
        self.filemenu.entryconfig(0,state=NORMAL)
        self.dt.setStop(True)
        self.stop_video_set_btns()
        self.master.update_idletasks()

    def stop_video_set_btns(self):
        self.start_btn.configure(state=NORMAL)  
        self.stop_btn.configure(state=DISABLED)    
        
    #def refresh_image(self):
    #    if self.filename_image != "" :
    #        self.clear_textbox()
    #        self.insert_image_to_frame(self.filename_image)        

    def camera_enable(self) :
        self.clear_youtube_text()
        self.camera_checkbtn.configure(state=DISABLED)
        self.fullfilename = ""
        if self.var_cbt.get() > -1:
            #print(f"amera_enable:{self.var_cbt.get()}")
            _camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)
            if  _camera.isOpened() == False:
                tk.messagebox.showerror("Error","Camera not open") 
                self.camera_checkbtn.deselect()
                self.filemenu.entryconfig(0,state=NORMAL)
                self.var_cbt.set(-1)
                _camera.release()
            else:
                self.start_stop(start=NORMAL, stop=DISABLED)
                #self.start_btn.configure(state=NORMAL)  
                #self.stop_btn.configure(state=DISABLED)
                self.filemenu.entryconfig(0,state=DISABLED)
                #self.refresh_btn.configure(state=DISABLED)
                _camera.release()                
        else:            
            #self.refresh_btn.configure(state=NORMAL)
            self.filemenu.entryconfig(0,state=NORMAL)
            self.start_stop(start=DISABLED, stop=DISABLED)
            #self.start_btn.configure(state=DISABLED)  
            #self.stop_btn.configure(state=DISABLED)
        self.camera_checkbtn.configure(state=NORMAL)
        return

    def saving_enable(self):
        self.nosave_image = False if self.var_saving_cbt.get() > -1 else True
        return

    def start_stop(self,start=DISABLED, stop=DISABLED):
        self.start_btn.configure(state=start)  
        self.stop_btn.configure(state=stop)

    def donothing(self):
        _filewin = Toplevel(self.master)
        _button = Button(_filewin, text="Do nothing button")
        _button.pack()

    def about(self):
        tk.messagebox.showinfo("About","image processing YOLOv5 V0.05\n2022.06.27\nTiszai Istvan\ntiszaii@hotmail.com") 

    def yolovx_rb_change(self):
        #self.algorithm.pause = True
        self.yolovX =  self.yolovx_switch(self.var_rdb.get())        
        self.setModel(yolovx=self.yolovX)
        self.var_saving_cbt.set(-1)
        #self.yolovX_prev = self.yolovX;
        if self.yolovX == YolovX.none :           
            self.yx_cbx.configure(state=DISABLED)
            self.model_load_btn.configure(state=DISABLED)
            self.save_checkbtn.configure(state=DISABLED)
            self.yolovX_model = None
            self.model=None
            self._youtube_set_btn.configure(state=DISABLED)
            #self.setModel()
        else:
            #_, _, filenames = next(walk("./weights"), (None, None, []))
            #self.yx_cbx['values'] = filenames
            self.yx_cbx.configure(state=NORMAL)
            self.model_load_btn.configure(state=NORMAL)
            self.save_checkbtn.configure(state=NORMAL)           
            self._youtube_set_btn.configure(state=NORMAL)
         #self.insert_image_to_frame( "../imagesVideos/640x480-azure-mist-solid-color-background.jpg", True)
        #print(f"rb :{self.yolovX}")

    def yolovx_switch(self, num):
        switcher = {
            0: YolovX.none,
            1: YolovX.yolov5,           
            }
        return switcher.get(num,"Invalid input")               

    def yx_cbx_changed(self, eventObject):        
        self.yolovX_model = self.var_cbx.get()        
        self.model_load_btn.configure(state=NORMAL)        

    def setModel(self, yolovx=YolovX.none) :
        #self.var_cbx.set('')
        if yolovx == YolovX.none:
            self.yx_cbx['values'] = ["None"]  
        else:        
            _, _, filenames = next(walk("./weights"), (None, None, []))
            filenames.insert(0,"None")
            self.yx_cbx['values'] = filenames
            #print("yolov5: not ready")
        self.yx_cbx.set("None")

    def model_load(self) :       
        if self.yolovX_model != None or self.yolovX_model != "None" :

            self.model_load_btn.configure(state=DISABLED)
            self.yx_cbx.configure(state=DISABLED)

            self.set_detectArgumentums_default()
            self.detectArgumentums['source']=self.filename_image
            self.detectArgumentums['nosave']=self.nosave_image
            self.dt=Detect(self, self.detectArgumentums)
            self.model = self.dt.set_modul(weight='weights/' + self.yolovX_model)

            self.model_load_btn.configure(state=NORMAL)
            self.yx_cbx.configure(state=NORMAL)

        else :
            self.model = None
        return

    def insert_textbox(self, message=""):        
        self.text_box.insert('end', message)

    def clear_textbox(self):        
        self.text_box.delete(1.0, 'end')
        
    def algoButtonsEnabled(self, enabled=True) :
        _state = NORMAL
        if enabled==False :
             _state = DISABLED        
        self.none_rdb.configure(state=_state)
        self.y1_rdb.configure(state=_state)
        if self.yolovX != YolovX.none :
            self.yx_cbx.configure(state=_state) 
            
    def clear_youtube_text(self):        
        self.youtube_text.delete(1.0, 'end')

    def youtube_set(self):     
        #https://www.youtube.com/watch?v=6ZudhqzpFco
        #https://www.youtube.com/watch?v=a5sYPWkwy9w&t=14s
        self.stop_btn.configure(state=DISABLED)
        videoName=self.youtube_text.get("1.0","end")
        if videoName != "\n" and videoName.find('youtube.com') > -1 :  
            if self.model == None :
                self.insert_textbox(message=f"Empty model,so not download:\n{videoName}")
            self.start_stop(start=NORMAL, stop=NORMAL)
            #self.start_btn.configure(state=NORMAL)  
            #self.stop_btn.configure(state=DISABLED)
            self.filename_image = videoName[:-1]
        else:

            self.start_btn.configure(state=DISABLED) 
            self.insert_textbox(message=f"Not 'youtube.com' link {videoName[:-1]}")
            
    def set_detectArgumentums_default(self) :
         self.detectArgumentums = {'weights':'weights/COCO-Detection/yolov5s.pt', # model path(s)
                                   'source':'', # file/dir/URL/glob, 0 for webcam
                                   'data':'data/coco128.yaml', # (optional) dataset.yaml path
                                   'img-size':[640], # inference size h,w
                                   'conf-thres':0.25, # confidence threshold
                                   'iou-thres':0.45, # NMS IoU threshold
                                   'max-det':1000, # maximum detections per image
                                   'device':'', # cuda device, i.e. 0 or 0,1,2,3 or cpu
                                   'view-img':True, # show results
                                   'save-txt':False, # save results to *.txt
                                   'save-conf':False, # save confidences in --save-txt labels
                                   'save-crop':False, # save cropped prediction boxes
                                   'nosave':True, # do not save images/videos
                                   'classes':None, # filter by class: --classes 0, or --classes 0 2 3
                                   'agnostic-nms':False, # class-agnostic NMS
                                   'augment':False, # augmented inference
                                   'visualize':False, # visualize features
                                   #'update':False, # update all models
                                   'project': 'outputs', # save results to project/name
                                   'line-thickness':3, # bounding box thickness (pixels)
                                   'hide-labels':False, # hide labels
                                   'hide-conf':False, # hide confidences
                                   'half':False, # use FP16 half-precision inference
                                   'model':None # model
                                   }

# return the center point
def canter(master, window_width, window_height ):
    _center_x = int(master.winfo_screenwidth()/2 - window_width/2)
    _center_y = int(master.winfo_screenheight()/2 - window_height/2)
    return ( _center_x, _center_y)