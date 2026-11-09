import argparse,json,time
from pathlib import Path
import cv2
from ultralytics import YOLO
from . import config
from .vision import polygon_pixels,appearance
from .tracker import ByteKalmanTracker
from .alerts import assess
from .drawing import draw_zone,draw_track,hud,banner
def main():
    p=argparse.ArgumentParser(description="Pool water-zone person monitoring and alert prototype")
    p.add_argument("--source",required=True,help="Video path or webcam index")
    p.add_argument("--model",default="yolov8n.pt");p.add_argument("--output",default="output/drowning_alert.mp4")
    p.add_argument("--events",default="output/events.json");p.add_argument("--conf",type=float,default=.35)
    p.add_argument("--device",default="",help="cpu, 0, cuda:0; blank uses Ultralytics auto")
    p.add_argument("--show",action="store_true");args=p.parse_args()
    source=int(args.source) if args.source.isdigit() else args.source
    model=YOLO(args.model);cap=cv2.VideoCapture(source)
    if not cap.isOpened():raise RuntimeError(f"Cannot open source: {args.source}")
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640;h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps<=0 or fps>240:fps=25.
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    if not writer.isOpened():raise RuntimeError(f"Cannot create output: {out}")
    event_path=Path(args.events);event_path.parent.mkdir(parents=True,exist_ok=True)
    poly=polygon_pixels(config.WATER_POLYGON,w,h)
    tracker=ByteKalmanTracker(config.MAX_MISSING_FRAMES,config.MIN_TRACK_HITS)
    events=[];alerted=set();idx=0;start=time.time()
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            idx+=1;kw=dict(source=frame,classes=[config.PERSON_CLASS_ID],conf=args.conf,imgsz=640,verbose=False)
            if args.device:kw["device"]=args.device
            result=model.predict(**kw)[0];dets=[];feats=[]
            if result.boxes is not None:
                for b in result.boxes:
                    box=b.xyxy[0].cpu().numpy().tolist();conf=float(b.conf[0].cpu().item())
                    dets.append({"box":box,"conf":conf});feats.append(appearance(frame,box))
            active=tracker.update(dets,feats,idx);draw_zone(frame,poly)
            counts={"SAFE":0,"NEAR_WATER":0,"IN_WATER":0,"POSSIBLE_DISTRESS":0};messages=[]
            for tid,t in active.items():
                state,lost=assess(t,poly,idx,fps);counts[state]+=1;draw_track(frame,t,state)
                event=None
                if state=="POSSIBLE_DISTRESS":event="POSSIBLE_DISTRESS"
                if lost:event="LOST_IN_WATER"
                if event and tid not in alerted:
                    alerted.add(tid);events.append({"frame":idx,"seconds":round(idx/fps,2),"track_id":tid,"event":event})
                    messages.append(f"ALERT! ID #{tid} {event.replace('_',' ')}")
            # Evaluate briefly missing tracks too, to catch a person disappearing in the pool zone.
            for t in tracker.tracks:
                if t.id not in active and t.missing/fps>=config.LOST_TRACK_SECONDS and t.state in ("IN_WATER","POSSIBLE_DISTRESS") and t.id not in alerted:
                    alerted.add(t.id);events.append({"frame":idx,"seconds":round(idx/fps,2),"track_id":t.id,"event":"LOST_IN_WATER"})
                    messages.append(f"ALERT! ID #{t.id} LOST IN WATER")
            elapsed=max(.001,time.time()-start);hud(frame,counts,idx/elapsed,len(alerted))
            if messages:banner(frame,messages[0])
            writer.write(frame)
            if args.show:
                cv2.imshow("Drowning Detection and Alert",frame);key=cv2.waitKey(1)&255
                if key==ord('q'):break
        event_path.write_text(json.dumps(events,indent=2),encoding="utf-8")
    finally:
        cap.release();writer.release();cv2.destroyAllWindows()
    print(f"Processed {idx}/{total} frames; alerts logged: {len(events)}")
    print(f"Video: {out.resolve()}");print(f"Events: {event_path.resolve()}")
    print("Experimental monitoring aid only; continuous human supervision is essential.")
if __name__=="__main__":main()
