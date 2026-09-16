# AI Use Log

This log records generative-AI assistance used during repository preparation.
AI output is treated as draft assistance, not as scientific authority. No
restricted patient data, secrets, API keys, credentials, or private files were
provided to the tool. The tool is not an author.

| date | tool/model | prompt purpose | output used | human verifier | verification evidence | change made |
|---|---|---|---|---|---|---|
| 2026-09-16 | GitHub Copilot | Prepare and audit Part 10 freeze artifacts | Repository inventory, artifact templates, and audit test candidates | Project user | Repository files, focused tests, smoke metadata, and freeze hashes were inspected | Additive freeze documentation only; scientific modules and protocol values were not changed |
| 2026-09-16 | GitHub Copilot | Audit Part 11 AI-use provenance and normalize the log schema | Documentation audit findings and the required table schema | Project user | Existing log, README, preregistration, freeze manifest, checksum manifest, and relevant diffs were inspected | Normalized this log to the required fields; no scientific implementation or freeze-status field was changed |

## Verification rule

Every AI-assisted factual claim, protocol value, source record, or code change
must be checked against repository evidence or an authoritative source by a
human verifier before confirmatory execution or manuscript submission. AI must
not invent citations, datasets, sample sizes, approvals, results, or clinical
claims; select favorable outcomes; redesign hypotheses or thresholds after test
inspection; or paraphrase sources that were not opened and verified.
