# Provenance workspace

The fetch tooling is tracked. Fetched source trees, downloaded artifacts, and
machine-specific revision records are ignored by Git.

Run source and model fetches independently:

```bash
.provenance/fetch_sources.sh
.provenance/fetch_models.sh
```

The manifest records source URL, revision selector, and declared license. The
script writes the resolved commit into `.provenance/records`. Model weights are
described by `.provenance/models.tsv`, downloaded into the ignored provenance
workspace, and never committed to FortAI.
