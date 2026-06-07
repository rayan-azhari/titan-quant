# Building a Production Quant Trading Stack, the book

A practitioner's guide to building, validating, deploying, and operating a
systematic trading system, using a real system ("Titan") as the running,
**sanitised** case study. Published as a navigable site (MkDocs Material) that
also exports to PDF.

> Audience: **public**. Everything here is the *process*, architecture,
> methodology, engineering, ops, and lessons. Proprietary specifics (exact
> parameters, instrument shortlists, live performance, account IDs, secrets)
> are deliberately omitted or replaced with clearly-labelled illustrative
> values. See [`docs/meta/style-and-redaction.md`](docs/meta/style-and-redaction.md).

## Build & preview

```bash
# from book/
python -m venv .venv && source .venv/bin/activate    # or: uv venv && source .venv/bin/activate
pip install -r requirements.txt                       # or: uv pip install -r requirements.txt
mkdocs serve            # live preview at http://127.0.0.1:8000
mkdocs build            # static site -> ./site
```

With `uv` and no venv:

```bash
uv run --with mkdocs-material --with pymdown-extensions mkdocs serve
```

## PDF

Uncomment `mkdocs-with-pdf` in `requirements.txt`, then `ENABLE_PDF_EXPORT=1 mkdocs build`.

## Status

Written incrementally. Each chapter file under `docs/` is either a full chapter
or a `Status: planned` outline. Chapter **6, "A backtest you can trust"** is
the completed reference sample that sets the voice and redaction bar; everything
else follows its template.

## Structure

`docs/index.md` is the preface. Chapters live in `docs/partN-*/`. Appendices in
`docs/appendix/`. The contributor/style guide is `docs/meta/style-and-redaction.md`.
The table of contents (and reading order) is defined by `nav:` in `mkdocs.yml`.
