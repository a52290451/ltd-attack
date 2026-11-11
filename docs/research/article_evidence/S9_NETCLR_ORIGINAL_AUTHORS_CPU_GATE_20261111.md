# NetCLR original-author CPU readiness and implementation-difference audit

Date: 11 November 2026. Target: SPIN-UMass original [repository](https://github.com/SPIN-UMass/Realistic-Website-Fingerprinting-By-Augmenting-Network-Traces), immutable commit \`c53ca10df0cad6bd5c8b27301a10dcc464278187\`.

## Verified directly in original notebooks

Files:
- \`artifacts/src/NetCLR/pre-training.ipynb\`
- \`artifacts/src/NetCLR/fine-tuning-cw.ipynb\`

Author pretraining:
- Defines \`DFNet(nn.Module)\`, \`DFsimCLR(nn.Module)\`, and \`Augmentor()\` directly in notebook code cells.
- \`DFNet\`: four paired Conv1D stages with BN/ELU-ReLU/pooling and 5,120-dimensional dense input.
- \`DFsimCLR\`: 5,120 -> 5,120 -> BN/ReLU -> 128 projection; augmentations produce two views.
- \`batch_size=256\`, \`temperature=0.5\`, Adam 0.0003, cosine scheduler and optional mixed precision.
- Notebook defines \`num_epoches=100\` as an initial variable, **but later passes \`num_epoches=401\` to its actual trainer**. The \`train\` method iterates \`range(self.num_epoches+1)\`, so executing unchanged would run **402 epochs indexed 0..401**, not simply 100. This must be reconciled with publication settings before source-faithful benchmark training.
- Pretraining source loads AWF-PT-sup; the LTD training must replace **only that population** by Historical DEV and audit every dependency.

Author fine-tuning:
- Reuses \`DFNet\` and \`DFsimCLR\` definitions. Its \`DFNet.weight_init\` adds a debugging \`print(n)\` absent from pre-training. This is not a learned-layer/topology difference.
- Classifier initialized by loading pretrained \`backbone.*\` tensors other than projection head \`backbone.fc.*\`.
- Uses random \`N\` traces per site, \`N=5\` in example, not LTD's all-observed-label Historical DEV chronological setup.
- Evaluates superior/inferior trace subsets; LTD will evaluate fixed 09B and Future-B populations **after C freeze**, not tune against them.

### Gated incompatibilities

1. Deprecated \`torch.nn.init.xavier_uniform(m.weight)\` call is mapped to \`xavier_uniform_(m.weight)\` **only inside verified extracted source in CPU smoke**, with exact occurrence count checked. This is a modern-PyTorch API compatibility fix; no pretrained weights change.
2. NetAugment's original burst-size \`OUTGOING_BURST_SIZE_CDF\`, \`outgoing_burst_sizes\`, \`bisect_left\` indexing and range-derived \`self.outgoing_burst_sizes\` require a deterministic distribution-alignment audit. The original notebook code is the source of truth; behavior changes must be recorded and justified separately.
3. \`S9_NETCLR_WFLIB_ADAPTED\` pretrain 100 epochs, batch 64 and locally adjusted augmentor is **not** author-original and cannot be relabeled. Preserve the checkpoint and all SHA evidence.
4. Before **original** NetCLR GPU training: frozen source-checkout PASS; original model forward/backward CPU PASS; original augmentation toy tests PASS; Historical DEV 57,916 coverage/65 labels and split PASS; pinned training hyperparameters; no external cohort access; independent output directory.
5. No original author-model training or accuracy is asserted by this document.

## CPU-only commands on Perseo

\`\`\`bash
cd ~/ltd-attack
git fetch origin feature/macro-v2-historical-only
git merge --ff-only FETCH_HEAD
bash scripts/sota/run_sota_original_cpu_only_v1.sh quick
bash scripts/sota/run_sota_original_cpu_only_v1.sh prepare-dev
\`\`\`

- \`quick\`: source SHA/pin, file inventory, original NetCLR source-cell extraction and original model CPU forward tests.
- \`prepare-dev\`: all the above plus Historical DEV direction cache of 57,916 examples. This is CPU/disk work and **not** contrastive pretraining.
- Neither stage invokes CUDA, accesses external 09B/Future-B, starts Dataset C, or writes model predictions.
