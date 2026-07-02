# data/

No raw data is committed to this repo, and there are no loader scripts here either -- each
training notebook calls `datasets.load_dataset(...)` directly (a one-line call with
notebook-specific schema handling, not worth factoring out). This file just documents what
gets pulled from HuggingFace at run time and where.

- Classification (`notebooks/01_train_classifier_rvl_cdip_mini.ipynb`): `dvgodoy/rvl_cdip_mini`
- Extraction/NER (`notebooks/02_train_extractor_funsd.ipynb`): `nielsr/funsd-layoutlmv3`

`naver-clova-ix/cord-v2` and `darentang/sroie` were candidate extraction datasets considered
early on but were not used for the extractor that actually shipped -- FUNSD alone was sufficient
to clear the project's own F1 bar. Revisit them only if extending the extractor beyond forms.
