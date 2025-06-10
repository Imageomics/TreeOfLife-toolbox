# Conventions

* Use n_ instead of num_ (e.g., n_imgs, n_bytes).
* Use bsz instead of batch_size in local loops or parameters.
* Avoid f-strings in logging statements: Prefer % formatting in logging (e.g., logger.info("Processed %d images", n_imgs)).
* @beartype.beartype on virtually every function and class for runtime checks.
