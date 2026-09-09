# scripts

Local development helpers. Nothing here runs on Kaggle — the Kaggle-side scripts live in
[`../kaggle/`](../kaggle/).

## `local_dry_run.ipynb`

Exercises every CPU stage on a laptop before it costs a Kaggle session. A hosted session
takes a commit cycle to iterate on; this takes seconds.

Nine sections. Three need no network at all — sections 3 (logic checks) and 9 (shard
round-trip) are the cheapest regression check in the project, so run them after any edit.

**It deliberately avoids torchcodec.** `datasets` 4.x decodes audio through torchcodec,
which needs a matching torch build *and* a system FFmpeg. The notebook casts to
`Audio(decode=False)` and decodes with `soundfile` instead. That has a second benefit: it
exercises the branch of `audio_io.decode_audio_field` that Kaggle does *not* take, so
between the two environments all the decode paths get covered.

`torch` is optional and only gates section 8 (VAD). Everything else runs without it.

### The token

Section 4 prompts via `getpass`, so it stays in memory and never reaches disk. On Kaggle
that one line becomes `UserSecretsClient().get_secret("HF_TOKEN")` — that is the only
auth difference.

**Before committing: Kernel → Restart & Clear Output.** The cell prints a length rather
than the token, but clearing is the habit worth keeping.

### Porting to Kaggle

The last markdown cell is a table of every difference. There are six, all mechanical.
