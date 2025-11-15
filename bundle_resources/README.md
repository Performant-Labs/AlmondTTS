# Bundle Resources

This directory is populated by `packaging/prepare_bundle_resources.sh` when building release artifacts. The script copies or downloads the XTTS model cache into `bundle_resources/models/`, which is ignored by git because the model files are large and derived artifacts.
