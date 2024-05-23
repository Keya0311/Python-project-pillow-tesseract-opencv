import zipfile
import PIL
from PIL import Image
import pytesseract
import cv2 as cv
import numpy as np

# loading the face detection classifier
face_cascade = cv.CascadeClassifier('readonly/haarcascade_frontalface_default.xml')

#extract all data and store them in to dictionary
def extract_data(file_path):
    info={}
    with zipfile.ZipFile(file_path,"r") as zfile:
        filelist=zfile.namelist()        
        for file in filelist:
            with zfile.open(file) as f:
            #read data as np array
                image = np.frombuffer(f.read(), np.uint8)
            #decode cv image 
                image=cv.imdecode(image,1)
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                faces=face_cascade.detectMultiScale(gray,1.3,5)
            
            #open image using PIL
                pil_image=Image.open(f)
                text=pytesseract.image_to_string(pil_image)                
            
            #store in dictionary
                info[file]=[pil_image, text, faces]
    return info      

#accept the string that matches information to the returned info of exctract_data fun
def search_images(string, info):
    keys=list(info.keys())
    for key in keys:
        if string in info[key][1]: #get text info
            image=info[key][0]      #image
            faces=info[key][2]      #with boxes
            
            #check for detected faces
            if len(faces)>0:
                #create contact_sheet
                contact_sheet_w=100
                contact_sheet_h=100
                rows = int(np.ceil(len(faces)/5))
                contact_sheet=PIL.Image.new(image.mode, (contact_sheet_w * 5, contact_sheet_h* rows))
                
                #crop each face using bounding box and paste image in to contact sheet
                r=0
                c=0
                for x,y,w,h in faces:
                    crop_img=image.crop((x,y,x+w,y+h))
                    if crop_img.width>contact_sheet_w:
                        crop_img=crop_img.resize((100, 100))
                        contact_sheet.paste(crop_img,(r,c))
                    else:
                        contact_sheet.paste(crop_img,(r,c))
                     
                    if crop_img.width+r == contact_sheet.width:
                        r=0
                        c=c+crop_img.height
                    else:
                        r += contact_sheet_w
                        
                print("Result found in file {}".format(key))
                display(contact_sheet)
            else:
                print("Results found in file {}".format(key))
                print("But there were no faces in that file!")     
                        
                    

test_1= extract_data("small_imge")
search_images("Christopher", test_1)                    