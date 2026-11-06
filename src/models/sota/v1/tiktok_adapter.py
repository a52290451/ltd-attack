"""Tik-Tok (Rahman et al., PETS 2020) directional timing DT representation.

A DT sample is direction_i * relative_timestamp_i, [1 x 5000].
This is NOT VarCNN's DT2 interarrival 2-channel representation.
The first timestamp is 0 so the first direction naturally maps to 0;
no invented offsets are inserted (paper-faithful DT information content).
The model architecture and optimization remain separately source-pinned.
"""
from __future__ import annotations
from array import array
import math

from .data_adapter import parse_direction
from .timing_adapter import parse_times

def tiktok_dt_from_sequences(directions:array,times:list[float],
                             length:int=5000)->array:
    if not isinstance(directions,array) or directions.typecode!="b":
        raise TypeError("expected array('b') packet directions")
    if not isinstance(times,list) or len(directions)!=len(times) or not times:
        raise ValueError("unpaired direction/time sequences")
    if type(length) is not int or length<1:
        raise ValueError("invalid DT length")
    if any(d not in (-1,1) for d in directions):
        raise ValueError("observed packet direction must be +/-1")
    if abs(times[0])>1e-3 or any(
         type(t) not in (float,int) or not math.isfinite(t) for t in times):
        raise ValueError("time_vector must be finite and start at zero")
    n=min(length,len(times))
    result=array("f",(float(directions[i])*float(times[i])
                      for i in range(n)))
    if len(result)<length:
        result.extend([0.]*(length-len(result)))
    return result

def tiktok_dt_from_text(direction_text:str,time_text:str,
                        length:int=5000)->array:
    return tiktok_dt_from_sequences(parse_direction(direction_text),
                                    parse_times(time_text),length)
