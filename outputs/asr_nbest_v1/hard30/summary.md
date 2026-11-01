# PhoASR N-best diagnostic — hard-30

## Generation

- Samples: 30 fixed IDs from `agent-architecture-audit-v1-val309/sample_ids.json`.
- Five candidates per sample: production greedy, beam width 3, beam width 5, and sampling at temperature 0.2 and 0.35 (top-p 0.9).
- Average/min/max candidates per sample: 5 / 5 / 5.
- Duplicate transcript rate: 56.7%; empty transcripts: 0; runtime failures: 0.
- Runtime: 423.7 seconds on the local RTX 4050.
- ASR score, average log probability, and no-speech probability are null because the backend did not expose reliable values.
- Cache, generation config, and manifest are in `outputs/asr_nbest_v1/`.

## ASR and downstream results

| Mode | WER | Tool F1 | State | Proxy | First full call | Exact trajectory |
|---|---:|---:|---:|---:|---:|---:|
| Current 1-best | 0.3048 | 0.1073 | 0.0000 | 0.0518 | 0.0000 | 0.0000 |
| WER oracle | 0.2890 | 0.1441 | 0.0000 | 0.0672 | 0.0000 | 0.0000 |
| Downstream oracle | — | — | — | — | — | — |
| Auto consensus | 0.2986 | 0.1116 | 0.0000 | 0.0518 | 0.0000 | 0.0000 |

N-best WER oracle improved 12/30 samples, was unchanged on 18/30, and worsened none. Oracle WER gain was 0.0158 absolute (5.2% relative); CER gain was 0.0126.

The mean WER over all five candidates was 0.3032. Mean per-sample best WER was 0.2890; mean per-sample best CER was 0.1652 (the WER-selected transcript's CER was 0.1657).

WER-oracle selection raised aggregate tool-call F1 by 0.0368 and proxy by 0.0154, but state match stayed at 0/30 and first full-call accuracy stayed at 0/30. Paired proxy outcomes were 6 improved, 5 worsened, and 19 unchanged. Auto consensus did not improve downstream proxy.

The ASR differences that matter are semantic: in `train_0571`, recovering “khóa lại” improves the lock request, while other alternatives still confuse payment intent, negation/conditionals, names, and organization terms. Some transcript changes did not help Qwen, and similar-WER candidates can produce different calls; lower WER alone does not select the best tool trajectory.

Downstream oracle was not run because evaluating all candidates requires multiple ToolEnv trajectories for each sample, conflicting with the competition requirement of exactly one ToolEnv per sample. The completed modes used one ToolEnv per sample.

## Interpretation

PhoASR's search space contains somewhat better transcripts, but the oracle WER gain is small. Better transcript selection improved call-level F1 on this hard subset without recovering a final state. Consensus selection was not useful. Candidate selection is a secondary opportunity; Qwen first-action grounding and multi-step planning remain the larger bottleneck. This result does not justify a more complex reranker or fusion experiment yet.
