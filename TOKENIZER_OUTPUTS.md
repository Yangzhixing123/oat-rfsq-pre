# Retained tokenizer training outputs

The local `output/` tree is intentionally ignored by Git. This project snapshot
retains only the following completed, date-prefixed action-tokenizer training
runs from the source workspace:

- `output/20260428/194619_train_oattok_libero10_N500` (559 MB)
- `output/20260710/123748_train_rfsq_oattok_libero10_N500` (471 MB)
- `output/20260807/115627_train_oattok_oat16_libero10_N500` (457 MB)
- `output/20260815/114643_train_rfsq4x2_oattok_libero10_N500` (639 MB)
- `output/20260818/192714_train_rfsq_oattok_rm-mt3_N600` (425 MB)
- `output/20260819/164626_train_oattok_robomimic_mt3_N600` (577 MB)
- `output/20260819/165102_train_rfsq_oattok_rm-mt3_N600` (710 MB)
- `output/20260825/113102_train_rfsq8_oattok_metaworld3_mw-mt4_N200` (480 MB)
- `output/20260825/113801_train_oat8_oattok_metaworld3_mw-mt4_N200` (479 MB)

Approximate retained size: 4.7 GB.

Policy-training, evaluation, comparison, reconstruction, and latency outputs
were not copied. Two empty/incomplete duplicate launch directories were also
excluded:

- `output/20260815/114644_train_rfsq4x2_oattok_libero10_N500`
- `output/20260818/192715_train_rfsq_oattok_rm-mt3_N600`
