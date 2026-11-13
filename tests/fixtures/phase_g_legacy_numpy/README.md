# Existing Study A checkpoint regression fixture

`legacy.json`/`legacy.pt` preserve an existing FIELD_ONLY verification-architecture checkpoint with one previously completed optimizer update, generated in the earlier, explicitly authorized tiny scheduling verification. No model was trained to create this compatibility fixture. A history scalar has the numerically identical NumPy float64 pickle representation to reproduce safe-load failure; tensor, optimizer, scheduler and RNG values are unchanged. The artifact retains its verified file/tree identities and original scientific/source metadata.

`original_run.json` records that earlier run's real diagnostic metadata. `fixed_data/` contains its frozen observations/query labels, regenerated once by the same unchanged reference recipe and checked against the checkpoint's **exact** fixed-data identity before archiving. No calibration, recertification or checkpoint-source relaxation was used. `provenance.json` records these details.

The exact-resume tests require the recorded Torch/device and numerical-runtime certificate, as the production loader does. Other scalar/security tests run independently of this platform fixture. This is verification data, not a research-trained model or generalization result.
