# Provenance workspace

The fetch tooling is tracked. Fetched source trees, downloaded artifacts, and
machine-specific revision records are ignored by Git.

Run:

```bash
.provenance/fetch_sources.sh
```

The manifest records source URL, revision selector, and declared license. The
script writes the resolved commit into `.provenance/records`. Model weights are
not fetched by this script and are never committed to FortAI.
