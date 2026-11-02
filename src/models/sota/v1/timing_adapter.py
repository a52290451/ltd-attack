"""Historical DEV direction+timing → WFlib VarCNN DT2 two-channel representation.

Source: src/data/01_preprocesamiento.py :: process_pcap_with_features
  - PcapReader TCP packet order, float(pkt.time), time[i]-time[0] seconds
  - direction[i] is +/- 1, aligned with the time of the same packet.
Upstream: WFlib/tools/data_processor.py at pinned WFlib
  X_dir = np.sign(X)
  X_time = np.diff(np.abs(X)); X_time[X_time < 0] = 0
  X_dir/time = length_align(..., 5000), 2 x 5000 input.

Crucial: If X=direction*time and time[0]=0, np.sign(X[0]) is
zero, losing the first direction. A +1 second offset to the signed
timestamp representation preserves sign while leaving all gaps identical:
  signed[i]=direction[i]*(offset+time[i]) with offset>abs(min(time))
  sign(signed[i])=direction[i], diff(abs(signed))=diff(time).
In keeping with WFlib's actual DT2 transform, any negative consecutive
timestamp deltas are set to zero, NOT sorted or rewritten. Count and report
negative steps separately to preserve scientific provenance.
We compute the two channels directly in Python float64 then store float32
only after differencing; no absolute epoch precision loss. For >=5001
packets, IAT at index 4999 needs the *5001st* timestamp (WFlib diffs
before trimming). No model training or external data access.
"""
from __future__ import annotations
from array import array
import ast
import math
from typing import NamedTuple

from .data_adapter import parse_direction

class DT2Example(NamedTuple):
    direction:array     # float32 encoded -1/+1 or zero padding
    interarrival:array  # float32 seconds >= 0
    original_packets:int
    input_length:int
    gaps_considered:int
    dropped_packets:int
    negative_gaps_total:int
    negative_gaps_used:int
    largest_backward_seconds:float

def parse_times(raw:str)->list[float]:
    if not isinstance(raw,str) or not raw.startswith("[") or not raw.endswith("]"):
        raise ValueError("time_vector must be a bracketed list")
    try:values=ast.literal_eval(raw)
    except (TypeError,ValueError,SyntaxError,RecursionError,MemoryError) as exc:
        raise ValueError("invalid time_vector") from exc
    if not isinstance(values,(list,tuple)) or not values:
        raise ValueError("time_vector must be a nonempty list")
    if any(type(t) not in (float,int) or not math.isfinite(t) for t in values):
        raise ValueError("time_vector must contain finite numeric seconds")
    result=[float(t) for t in values]
    if abs(result[0])>1e-3:
        raise ValueError("expected per-PCAP relative times starting at zero")
    # PCAP frame order can contain decreasing timestamps. The WFlib DT2
    # reference does not sort them; it clips negative *differences* to zero.
    # We preserve original timestamps and expose negative-gap diagnostics.
    return result

def dt2_from_sequences(directions:array,times:list[float],length:int=5000)->DT2Example:
    if not isinstance(directions,array) or directions.typecode!="b":
        raise TypeError("directions must be array('b')")
    if type(length) is not int or length<=0:
        raise ValueError("length must be a positive integer")
    n=len(directions)
    if n==0 or len(times)!=n:
        raise ValueError("direction/time packet lengths differ or are empty")
    if any(d not in (-1,1) for d in directions):
        raise ValueError("observed packet direction must be +/-1, not padding")
    if not all(type(x) in (float,int) and math.isfinite(x) for x in times):
        raise ValueError("time entries must be finite")
    if abs(times[0])>1e-3:
        raise ValueError("relative time starts away from zero")
    # WFlib DT2, data_processor.py: diff(abs(X)); negative differences -> 0.
    # Never sort timestamps, delete packets, silently re-time observations or
    # infer missing time. Diagnostics make each clipping event observable.
    backward=[max(0.0,float(times[i])-float(times[i+1]))
              for i in range(n-1)]
    negative_total=sum(x>0 for x in backward)
    largest_back=max(backward,default=0.0)
    n_dir=min(n,length)
    n_gaps=min(n-1,length)
    c_dir=array("f",(float(directions[i]) for i in range(n_dir)))
    c_time=array("f",(float(max(0.0,times[i+1]-times[i]))
                      for i in range(n_gaps)))
    if len(c_dir)<length:c_dir.extend([0.0]*(length-len(c_dir)))
    if len(c_time)<length:c_time.extend([0.0]*(length-len(c_time)))
    if len(c_dir)!=length or len(c_time)!=length:
        raise AssertionError("DT2 channels did not align to requested length")
    return DT2Example(c_dir,c_time,n,length,n_gaps,max(0,n-length),
                      negative_total,sum(x>0 for x in backward[:n_gaps]),
                      largest_back)

def dt2_from_text(direction_text:str,time_text:str,length:int=5000)->DT2Example:
    d=parse_direction(direction_text)
    t=parse_times(time_text)
    return dt2_from_sequences(d,t,length)

def reference_signed_timestamps(directions:array,times:list[float])->list[float]:
    """Diagnostic only: prove equivalence to WFlib sign/diff(abs(.)) recipe."""
    if len(directions)!=len(times) or not directions:
        raise ValueError("unaligned diagnostic vectors")
    if any(x not in (-1,1) for x in directions):
        raise ValueError("signed time needs observed +/-1 directions")
    if any(type(t) not in (int,float) or not math.isfinite(t) for t in times):
        raise ValueError("nonfinite signed timestamp")
    offset=1.0+max(0.0,-min(times))
    return [float(directions[i])*(float(times[i])+offset)
            for i in range(len(times))]
