import cv2
COL={"SAFE":(255,220,0),"NEAR_WATER":(0,220,255),"IN_WATER":(0,165,255),"POSSIBLE_DISTRESS":(0,0,255)}
def draw_zone(frame,poly):
    over=frame.copy();cv2.fillPoly(over,[poly],(255,120,0));cv2.addWeighted(over,.20,frame,.80,0,frame)
    cv2.polylines(frame,[poly],True,(255,180,0),2,cv2.LINE_AA)
    x,y=poly[0];cv2.putText(frame,"WATER ZONE",(x+8,y+22),0,.6,(255,255,255),2)
def draw_track(frame,t,state):
    x1,y1,x2,y2=map(int,t.box);c=COL[state]
    cv2.rectangle(frame,(x1,y1),(x2,y2),c,2,cv2.LINE_AA)
    label=f"ID #{t.id} | {state.replace('_',' ')}"
    (tw,th),_=cv2.getTextSize(label,0,.48,1);top=max(0,y1-th-9)
    cv2.rectangle(frame,(x1,top),(x1+tw+8,y1),c,-1)
    cv2.putText(frame,label,(x1+4,max(th+2,y1-4)),0,.48,(0,0,0),1,cv2.LINE_AA)
    pts=[p for _,p in list(t.history)[-20:]]
    for a,b in zip(pts,pts[1:]):cv2.line(frame,a,b,c,2,cv2.LINE_AA)
def hud(frame,counts,fps,alerts):
    h,w=frame.shape[:2];cv2.rectangle(frame,(12,12),(min(w-12,390),100),(15,18,24),-1)
    cv2.putText(frame,"DROWNING DETECTION AND ALERT",(22,38),0,.55,(0,255,0),2)
    cv2.putText(frame,f"SAFE {counts['SAFE']}  NEAR WATER {counts['NEAR_WATER']}",(22,63),0,.45,(255,255,255),1)
    cv2.putText(frame,f"IN WATER {counts['IN_WATER']}  ALERTS {alerts}  FPS {fps:.1f}",(22,86),0,.42,(0,210,255),1)
def banner(frame,text):
    h,w=frame.shape[:2];cv2.rectangle(frame,(0,110),(w,148),(0,0,220),-1)
    cv2.putText(frame,text,(12,136),0,.62,(255,255,255),2)
