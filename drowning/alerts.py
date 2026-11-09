from .vision import center,inside,poly_distance
from . import config
def assess(track,poly,frame_idx,fps):
    x1,y1,x2,y2=track.box
    foot=(int((x1+x2)/2),int(y2))
    in_water=inside(foot,poly);near=(not in_water and poly_distance(foot,poly)<=config.NEAR_WATER_PIXELS)
    c=center(track.box)
    motion=0
    if len(track.history)>1:
        old=track.history[-2][1];motion=((c[0]-old[0])**2+(c[1]-old[1])**2)**.5
    track.count_in_water=track.count_in_water+1 if in_water else 0
    track.low_motion=track.low_motion+1 if in_water and motion<2 else 0
    if in_water and track.low_motion>=int(4*fps):state="POSSIBLE_DISTRESS"
    elif in_water:state="IN_WATER"
    elif near:state="NEAR_WATER"
    else:state="SAFE"
    track.state=state
    lost_in_water=track.missing/fps>=config.LOST_TRACK_SECONDS and state in ("IN_WATER","POSSIBLE_DISTRESS")
    return state,lost_in_water
