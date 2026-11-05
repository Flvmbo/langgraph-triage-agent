# Eval results (hard set)

16 tickets per model.

| Model | Category | Priority | Action | All three | Errors | Mean latency | Tokens |
|---|---|---|---|---|---|---|---|
| claude | 100% | 100% | 100% | 100% | 0/16 | 9.8s | 104,413 |
| claude-haiku | 100% | 100% | 100% | 100% | 0/16 | 7.8s | 109,415 |
| claude-opus | 100% | 94% | 100% | 94% | 0/16 | 11.4s | 93,293 |

**All three correct, by trap type**

| Model | boundary | evidence | identity | injection | language | multi-issue | over-application | Failed tickets |
|---|---|---|---|---|---|---|---|---|
| claude | 4/4 | 2/2 | 1/1 | 2/2 | 2/2 | 3/3 | 2/2 | none |
| claude-haiku | 4/4 | 2/2 | 1/1 | 2/2 | 2/2 | 3/3 | 2/2 | none |
| claude-opus | 4/4 | 2/2 | 1/1 | 1/2 | 2/2 | 3/3 | 2/2 | H06 |
