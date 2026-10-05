# LockStep: Replication Package

This package accompanies the paper *LockStep*, which detects low-rate backdoor
poisoning of neural code models from the learning curves of training samples.
It contains (1) the code to reproduce the experiments and (2) the results of
RQ1-RQ3.

`data/README.md` lists the source of each dataset and
how the poisoned datasets are built.

## Layout

```
data/          dataset preparation and trigger generation (see data/README.md)
training/      recording runs: fine-tune CodeT5+ on the poisoned training sets
               and record the per-epoch probability of every training sample
lockstep/      LockStep (Section 4)
baselines/     OptiSS and KillBadCode
retraining/    retraining on the cleaned data and evaluation (ASR, FTR, clean
               performance) of CodeT5+ (RQ2) and StarCoderBase-1B (RQ3)
results/       results of RQ1-RQ3
requirements.txt
```

## Setup

```bash
conda create -n lockstep python=3.10 && conda activate lockstep
pip install -r requirements.txt
```

The models are downloaded from Hugging Face (`Salesforce/codet5p-220m`,
`Salesforce/codet5p-770m`, `bigcode/starcoderbase-1b`). KillBadCode also
needs KenLM (https://github.com/kpu/kenlm). All scripts are run from the
package root.

## Running the experiments

Commands are shown for one task and trigger. The tasks are `summarization`,
`repair`, and `completion`; the triggers are `fix`, `grammar`, and `llm`
(fixed, grammar, and LLM in the paper and results).

1. **Data.** Build the datasets as described in `data/README.md`.

2. **Recording runs.** Train CodeT5+ on the poisoned training sets and record
   the training dynamics. These runs are also the undefended models of RQ2.

   ```bash
   bash training/run_codet5p.sh summarization fix 11 0  # TASK TRIGGER SEED GPU (one run)
   bash training/test_codet5p.sh summarization fix 0    # ASR of the undefended models
   ```

   The seeds of the 90 runs are in the `seed` column of `results/*_per_run.csv`.

3. **RQ1: Detection.** Run the three defenses on every poisoned training set:
   LockStep (on the training dynamics recorded in step 2), and the baselines
   OptiSS and KillBadCode. Each defense writes the samples it removes to
   `detected_indices_<defense>.json` in each run directory, which steps 4 and
   5 read.

   ```bash
   bash lockstep/run_lockstep.sh                        # LockStep, all tasks
   LMPLZ_PATH=/path/to/kenlm/build/bin/lmplz \
     bash baselines/run_baselines.sh summarization 0    # OptiSS and KillBadCode, TASK GPU
   ```

4. **RQ2: Retraining CodeT5+ on the cleaned data.** For each defense, remove
   the detected samples from each poisoned training set and train CodeT5+
   again from its pre-trained weights, with the same configuration and seed as
   the recording run. Each defended model is compared with its undefended
   model (the recording run) on ASR, clean performance, and FTR.

   ```bash
   bash retraining/eval_undefended_codet5p.sh summarization fix 0          # clean performance and FTR of the undefended models
   bash retraining/run_retrain_codet5p.sh summarization fix lockstep 0     # retrain and evaluate; lockstep | optiss | killbadcode
   ```

5. **RQ3: StarCoderBase-1B as the deployed model.** The recording model stays
   CodeT5+: the removed samples are the ones detected with CodeT5+ in step 3,
   so the cleaned data is the same as in RQ2. StarCoderBase-1B is fine-tuned on
   the poisoned data (undefended) and on the data cleaned by each defense, and
   evaluated as in RQ2. Unlike CodeT5+, the undefended StarCoderBase-1B models
   do not exist yet, so `undefended` trains them on the poisoned data and then
   evaluates them.

   ```bash
   bash retraining/run_retrain_starcoder.sh summarization fix undefended 0  # train and evaluate the undefended models
   bash retraining/run_retrain_starcoder.sh summarization fix lockstep 0    # lockstep | optiss | killbadcode
   ```

   In steps 4 and 5, the ASR, clean performance, and FTR of each model are
   written to `metrics.json` in its output directory.

## Results

All rates are in %. In each run, 9 of the 10,000 training samples (0.09%) are
poisoned. Settings are the 9 combinations of task (CS, CR, CM) and trigger
(fixed, grammar, LLM), with 10 runs each.

| File | Paper | Content |
|---|---|---|
| `results/rq1_detection.csv` | RQ1, detection table | recall and FPR of each defense per setting and per task (`all`); `p_binomial`: one-sided binomial test against random removal |
| `results/rq1_detection_per_run.csv` | RQ1 | removed samples, poisoned samples removed, recall, and FPR of each defense in each of the 90 runs |
| `results/rq2_asr.csv` | RQ2, ASR table | CodeT5+: poisoned samples left and ASR per setting and defense; `p_wilcoxon`: paired Wilcoxon signed-rank test against Undefended |
| `results/rq2_clean.csv` | RQ2, clean-performance table | CodeT5+: clean performance and FTR averaged over the 30 runs of each task; number of triggers with a significant decrease |
| `results/rq2_per_run.csv` | RQ2 | CodeT5+: ASR, clean performance, and FTR of each of the 360 models (90 undefended, 270 defended) |
| `results/rq3_asr.csv` | RQ3, ASR table | StarCoderBase-1B, same as `rq2_asr.csv` |
| `results/rq3_clean.csv` | RQ3, clean-performance table | StarCoderBase-1B, same as `rq2_clean.csv` |
| `results/rq3_per_run.csv` | RQ3 | StarCoderBase-1B, same as `rq2_per_run.csv` |
