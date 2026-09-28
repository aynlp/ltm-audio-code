# Anonymous Release Checklist

- Start from a fresh repository, not the original experiment checkout or its
  Git history.
- Track only the files in this release tree. Do not add data, audio, token
  arrays, model weights, caches, checkpoints, predictions, logs, figures, or
  generated result files.
- Remove usernames, personal directories, hostnames, e-mail addresses,
  credentials, experiment tracker links, and private URLs from every tracked
  file and commit message.
- Do not add contributor metadata, a citation file, or badges that disclose an
  account or organization during double-blind review.
- Run `python scripts/audit_release.py --root .` and inspect all tracked files
  before uploading.
- Upload the clean repository to a newly created anonymous account/repository,
  then create the review link from that repository rather than from a fork or
  an existing project with history.
