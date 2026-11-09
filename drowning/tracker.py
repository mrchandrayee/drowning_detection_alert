import cv2, numpy as np
from collections import deque
from .vision import center,iou,similarity
def make_kalman(box):
    k=cv2.KalmanFilter(8,4)
    k.transitionMatrix=np.array([[1,0,0,0,1,0,0,0],[0,1,0,0,0,1,0,0],
      [0,0,1,0,0,0,1,0],[0,0,0,1,0,0,0,1],[0,0,0,0,1,0,0,0],
      [0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]],np.float32)
    k.measurementMatrix=np.zeros((4,8),np.float32);k.measurementMatrix[:,:4]=np.eye(4,dtype=np.float32)
    k.processNoiseCov=np.eye(8,dtype=np.float32)*.01;k.measurementNoiseCov=np.eye(4,dtype=np.float32)*.1
    k.errorCovPost=np.eye(8,dtype=np.float32)
    x1,y1,x2,y2=map(float,box)
    k.statePost=np.array([[(x1+x2)/2],[(y1+y2)/2],[max(1,x2-x1)],[max(1,y2-y1)],[0],[0],[0],[0]],np.float32)
    return k
class Track:
    def __init__(self,tid,det,feat,frame):
        self.id=tid;self.kf=make_kalman(det["box"]);self.box=det["box"]
        self.appearance=feat;self.hits=1;self.missing=0;self.last=frame
        self.history=deque([(frame,center(self.box))],maxlen=40)
        self.state="SAFE";self.low_motion=0;self.count_in_water=0
    def predict(self):
        p=self.kf.predict().flatten();cx,cy,bw,bh=p[:4]
        self.box=[cx-max(1,bw)/2,cy-max(1,bh)/2,cx+max(1,bw)/2,cy+max(1,bh)/2]
        self.missing+=1
    def correct(self,det,feat,frame):
        x1,y1,x2,y2=det["box"]
        z=np.array([[(x1+x2)/2],[(y1+y2)/2],[max(1,x2-x1)],[max(1,y2-y1)]],np.float32)
        self.kf.correct(z);self.box=det["box"];self.hits+=1;self.missing=0;self.last=frame
        if feat is not None:
            if self.appearance is None:self.appearance=feat
            else:
                self.appearance=.85*self.appearance+.15*feat
                n=np.linalg.norm(self.appearance)
                if n>1e-8:self.appearance/=n
        self.history.append((frame,center(self.box)))
class ByteKalmanTracker:
    # ByteTrack-inspired two-stage confidence matching with appearance re-identification.
    def __init__(self,max_missing=45,min_hits=2):
        self.tracks=[];self.next_id=1;self.max_missing=max_missing;self.min_hits=min_hits
    def update(self,dets,features,frame):
        for t in self.tracks:t.predict()
        pairs=[]
        for ti,t in enumerate(self.tracks):
            for di,d in enumerate(dets):
                ov=iou(t.box,d["box"]);sim=similarity(t.appearance,features[di])
                c1=center(t.box);c2=center(d["box"]);dist=((c1[0]-c2[0])**2+(c1[1]-c2[1])**2)**.5
                scale=max(30,(d["box"][2]-d["box"][0])*.8,(d["box"][3]-d["box"][1])*.8)
                if ov>.05 or (dist<scale and sim>.55):
                    pairs.append((.58*(1-ov)+.42*(1-sim),ti,di))
        used_t=set();used_d=set()
        for _,ti,di in sorted(pairs):
            if ti in used_t or di in used_d:continue
            self.tracks[ti].correct(dets[di],features[di],frame)
            dets[di]["track_id"]=self.tracks[ti].id;used_t.add(ti);used_d.add(di)
        for di,d in enumerate(dets):
            if di not in used_d and d["conf"]>=.45:
                t=Track(self.next_id,d,features[di],frame);d["track_id"]=self.next_id
                self.next_id+=1;self.tracks.append(t)
        self.tracks=[t for t in self.tracks if t.missing<=self.max_missing]
        return {t.id:t for t in self.tracks if t.hits>=self.min_hits and t.missing<=2}
