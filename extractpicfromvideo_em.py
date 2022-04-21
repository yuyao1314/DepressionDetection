
import cv2
import os
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
def cutvideo(datapath,savepath):
    vc = cv2.VideoCapture(datapath) 
    c = 1
    count=0
    if vc.isOpened(): 
        rval, frame = vc.read()
        # print('yes!!!')
    else:
        rval = False
    timeF = 1  
    while rval:
        rval, frame = vc.read()
       
        if (c % timeF == 0 ):  
            if frame is not None:
                cv2.imencode('.jpg', frame)[1].tofile(savepath+'/' + date[i]+'{0}'.format(j)+str(count) + '.jpg')
                count+=1
        c = c + 1
        cv2.waitKey(1)
    vc.release()
   


if __name__ == '__main__':
    parent_path = "/home/som/lab-data/avce2013/Training/Training_Videos/"
    root_path = '/home/som/lab-data/yuyaodata/2013test/'

    date = os.listdir(parent_path)
    for i in range(len(date)):
        datafolder = os.path.join(parent_path, date[i], 'physiology','surprised')
        print("AAAAAA:",datafolder)
        vonlunteers = os.listdir(datafolder)
        for j in range(len(vonlunteers)):
            videopath = datafolder + '/' + vonlunteers[j] + '/' + 'visible.MP4'

            save_path = os.path.join(root_path,'surprised')
            print(videopath)
            cutvideo(videopath, save_path)











