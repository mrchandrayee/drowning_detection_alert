import cv2, numpy as np
def polygon_pixels(points,w,h):
    return np.array([(int(x*w),int(y*h)) for x,y in points],np.int32)
def inside(point,poly):
    return cv2.pointPolygonTest(poly,(float(point[0]),float(point[1])),False)>=0
def poly_distance(point,poly):
    return abs(cv2.pointPolygonTest(poly,(float(point[0]),float(point[1])),True))
def center(box):
    x1,y1,x2,y2=box
    return (int((x1+x2)/2),int((y1+y2)/2))
def iou(a,b):
    ax,ay,bx,by=a; cx,cy,dx,dy=b
    ix1,iy1=max(ax,cx),max(ay,cy); ix2,iy2=min(bx,dx),min(by,dy)
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    aa=max(0,bx-ax)*max(0,by-ay); bb=max(0,dx-cx)*max(0,dy-cy)
    return inter/(aa+bb-inter+1e-9)
def appearance(frame,box,bins=32):
    h,w=frame.shape[:2]; x1,y1,x2,y2=map(int,box)
    crop=frame[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
    if crop.size==0:return None
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    a=cv2.calcHist([hsv],[0],None,[bins],[0,180]).flatten()
    b=cv2.calcHist([hsv],[1],None,[bins],[0,256]).flatten()
    f=np.r_[a,b].astype(np.float32); n=np.linalg.norm(f)
    return f/n if n>1e-8 else f
def similarity(a,b):
    if a is None or b is None:return 0.
    return float(np.clip(np.dot(a,b),0,1))
