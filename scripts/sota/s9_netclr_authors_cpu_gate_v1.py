#!/usr/bin/env python3
"""CPU-only original-author NetCLR model and augmentation source compatibility gate.

Uses ONLY code cells from pinned SPIN-UMass authors' notebook; no WFlib model,
no GPU, no data acquisition, no model training, and no scientific output.
The only permitted source compatibility patch is deprecated
torch.nn.init.xavier_uniform -> xavier_uniform_ (tracked and asserted).

--stage inspect: stdlib-only audit of original notebook cells and overrides.
--stage smoke: also instantiate original-author DFNet and DFsimCLR and execute
               small CPU-only forward/backward and NetAugment smoke.
"""
from __future__ import annotations
import argparse
import ast
import bisect
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.sota.sota_paths_v1 import ORIGINAL
REPO=ORIGINAL/"S9_NetCLR_Authors"
COMMIT="c53ca10df0cad6bd5c8b27301a10dcc464278187"
PRE=REPO/"artifacts/src/NetCLR/pre-training.ipynb"
FIN=REPO/"artifacts/src/NetCLR/fine-tuning-cw.ipynb"
MODEL_CLASSES=("DFNet","DFsimCLR")
AUGMENT_CLASS="Augmentor"
RULES={
    "allowed_patches":["torch.nn.init.xavier_uniform -> torch.nn.init.xavier_uniform_"],
    "batch_size_original":256,
    "contrastive_temperature_original":0.5,
    "pretraining_iterations_original":401,
    "pretraining_source":"authors' notebook / AWF-PT-sup source, to be replaced by Historical DEV",
    "selection":"DEV_LATE only, never external cohorts",
}
def must(ok,msg):
    if not ok: raise RuntimeError("S9_AUTHOR_GATE_BLOCKED "+msg)

def notebook(p):
    must(p.is_file(),"missing notebook "+str(p))
    payload=json.loads(p.read_text(encoding="utf-8"))
    must(isinstance(payload.get("cells"),list),"notebook cells missing")
    return ["".join(c["source"]) if isinstance(c.get("source"),list)
            else str(c.get("source",""))
            for c in payload["cells"] if c.get("cell_type")=="code"]

def extract_class(cells,name):
    for idx,txt in enumerate(cells):
        try:tree=ast.parse(txt)
        except SyntaxError:continue
        classes=[node for node in tree.body
                 if isinstance(node,ast.ClassDef) and node.name==name]
        if len(classes)==1:
            cls=classes[0]
            segment=ast.get_source_segment(txt,cls)
            must(bool(segment),name+" source extraction failed")
            return idx,segment
    raise RuntimeError("S9_AUTHOR_GATE_BLOCKED missing upstream class "+name)

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for part in iter(lambda:f.read(1024*1024),b""):
            h.update(part)
    return h.hexdigest()

def inspect():
    must((REPO/".git").exists(),"pinned authors repository absent")
    import subprocess
    head=subprocess.run(["git","-C",str(REPO),"rev-parse","HEAD"],
                        check=True,capture_output=True,text=True).stdout.strip()
    must(head==COMMIT,"authors source commit mismatch")
    dirty=subprocess.run(["git","-C",str(REPO),"status","--porcelain",
                          "--untracked-files=no"],
                         check=True,capture_output=True,text=True).stdout.strip()
    must(not dirty,"authors code tracked files modified")
    pre=notebook(PRE)
    fin=notebook(FIN)
    extracted={name:extract_class(pre,name) for name in
               (*MODEL_CLASSES,AUGMENT_CLASS)}
    fine={name:extract_class(fin,name) for name in MODEL_CLASSES}
    # The authors' fine-tuning notebook adds an active diagnostic print(n)
    # inside DFNet.weight_init; remove ONLY this debug statement for parity
    # comparison, not from the implementation executed in the smoke.
    def normalized_ast(source,allow_debug=False):
        tree=ast.parse(source)
        if allow_debug:
            class RemoveKnownDebug(ast.NodeTransformer):
                def visit_Expr(self,node):
                    if (isinstance(node.value,ast.Call)
                        and isinstance(node.value.func,ast.Name)
                        and node.value.func.id=="print"
                        and len(node.value.args)==1
                        and isinstance(node.value.args[0],ast.Name)
                        and node.value.args[0].id=="n"):
                        return None
                    return self.generic_visit(node)
            tree=RemoveKnownDebug().visit(tree)
            ast.fix_missing_locations(tree)
        return ast.dump(tree,include_attributes=False)
    for name in MODEL_CLASSES:
        must(normalized_ast(extracted[name][1])
             ==normalized_ast(fine[name][1],allow_debug=True),
             "authors pretrain and fine-tuning topology drift "+name)
    all_pre="\n".join(pre)
    must("batch_size = 256" in all_pre and "temperature = 0.5" in all_pre,
         "original notebook hyperparameter contract changed")
    must("num_epoches = 401" in all_pre,
         "authors actual loop-length contract changed")
    must("xavier_uniform(" in extracted["DFNet"][1],
         "unexpected upstream initialization API, review patch")
    model_src="\n\n".join(extracted[n][1] for n in MODEL_CLASSES)
    must(model_src.count("torch.nn.init.xavier_uniform(")==1,
         "expected single compatibility patch")
    compiled_src=model_src.replace(
        "torch.nn.init.xavier_uniform(",
        "torch.nn.init.xavier_uniform_(")
    report={
      "classification":"S9_AUTHOR_NETCLR_NOTEBOOK_CPU_SOURCE_AUDIT",
      "repo":str(REPO),"commit":COMMIT,
      "pretrain_notebook_sha256":sha(PRE),
      "finetune_notebook_sha256":sha(FIN),
      "model_cell_indices":{k:extracted[k][0] for k in MODEL_CLASSES},
      "augmentor_cell_index":extracted[AUGMENT_CLASS][0],
      "source_model_ast_equivalent_ignoring_finetune_debug_print":True,
      "source_compatibility_fix_count":1,"rules":RULES,
      "source_used":"authors_original, not WFlib",
      "runtime_training_performed":False,
      "external_cohorts_opened":False,
      "dataset_c_opened":False,
    }
    return report,compiled_src,extracted[AUGMENT_CLASS][1]

def smoke(model_src,aug_src,report):
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    # Only class-def AST nodes from the pinned, SHA-checked authors' notebook:
    # no notebook data-download, fit, model checkpoint, file or external code cells.
    namespace={"torch":torch,"nn":nn,"F":F,"np":np,"random":random,
               "bisect":bisect}
    model_ast=ast.parse(model_src)
    must(all(isinstance(n,ast.ClassDef) for n in model_ast.body),
         "unexpected executable notebook statements")
    exec(compile(model_ast,"s9_author_model_cells","exec"),namespace)
    torch.manual_seed(42)
    m=namespace["DFsimCLR"](namespace["DFNet"](out_dim=512),out_dim=128)
    m.eval()
    with torch.inference_mode():
        z=m(torch.zeros(2,1,5000))
    must(z.shape==(2,128) and bool(torch.isfinite(z).all().item()),
         "authors SimCLR CPU forward failed")
    # 65-class supervised classifier instantiated from the same original DFNet.
    clf=namespace["DFNet"](out_dim=65)
    clf.eval()
    with torch.inference_mode():
        y=clf(torch.zeros(2,1,5000))
    must(y.shape==(2,65) and bool(torch.isfinite(y).all().item()),
         "authors supervised CPU forward failed")
    print("S9_AUTHOR_MODEL_SMOKE_PASS",
          json.dumps({"contrastive_output":list(z.shape),
                      "classification_output":list(y.shape),
                      "original_notebook_sha256":report["pretrain_notebook_sha256"]}),
          flush=True)
    # Original NetAugment class may reference notebook global arrays/CDF.
    # We inspect its signatures and attributes here; synthetic behavior is a
    # separate gate because the original notebook's global burst-size sampling
    # has an unresolved CDF index alignment that must not be silently patched.
    aug_ast=ast.parse(aug_src)
    must(len(aug_ast.body)==1 and isinstance(aug_ast.body[0],ast.ClassDef),
         "unexpected Augmentor code statement")
    methods={f.name for f in aug_ast.body[0].body
             if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef))}
    needed={"find_bursts","increase_incoming_bursts",
            "decrease_incoming_bursts","merge_incoming_bursts",
            "add_outgoing_burst","create_trace_from_burst_sizes",
            "shift","augment"}
    must(needed<=methods,"authors augmentation function set changed")
    print("S9_AUTHOR_AUGMENT_SIGNATURE_PASS",sorted(methods),flush=True)
    # Synthetic-only execution of the ORIGINAL author Augmentor. Construct the
    # authors' globals from a known distribution; do not inject WFlib's fix.
    # This checks compatibility, not scientific augmentation fidelity.
    burst_samples=[1,2,3,4]*100
    max_outgoing_burst_size=max(burst_samples)
    counts,bins=np.histogram(burst_samples,bins=max_outgoing_burst_size-1)
    pdf=counts/counts.sum()
    cdf=np.zeros_like(bins)
    cdf[1:]=np.cumsum(pdf)
    namespace.update({
       "outgoing_burst_sizes":burst_samples,
       "max_outgoing_burst_size":max_outgoing_burst_size,
       "OUTGOING_BURST_SIZE_CDF":cdf,
    })
    exec(compile(aug_ast,"s9_author_original_augmentor","exec"),namespace)
    random.seed(42)
    np.random.seed(42)
    aug=namespace["Augmentor"]()
    seq=np.zeros(5000,dtype=np.int32)
    seq[:250]=np.asarray(([1]*35+[-1]*40+[1]*25+[-1]*30)*2)[:250]
    for _ in range(12):
        out=np.asarray(aug.augment(seq.copy()))
        must(out.shape==(5000,) and
             np.all(np.isin(out,[-1,0,1])),
             "original authors' augmentation synthetic output")
    print("S9_AUTHOR_AUGMENT_SYNTHETIC_CPU_PASS",
          "augmented_traces",12,"original_CDF_bins",len(cdf),flush=True)
    print("S9_AUTHOR_AUGMENT_FIDELITY_AUDIT_PENDING",
          "source-level sample distribution vs Historical DEV not yet validated")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=("inspect","smoke"),required=True)
    args=ap.parse_args()
    report,model,augment=inspect()
    print("S9_AUTHOR_NETCLR_SOURCE_PASS",json.dumps(report,indent=2),
          flush=True)
    if args.stage=="smoke":smoke(model,augment,report)

if __name__=="__main__":main()
