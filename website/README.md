# VisualMontage website

**Manipulating the Beliefs of Vision-Language Agents with Authentic Image Sequences**

[Project page](https://CharlesJW222.github.io/VisualMontage/)

VisualMontage studies visual cognitive collusion: authentic images and locally faithful captions can jointly induce a conclusion contradicted by the complete source context. This directory contains the research project page.

## News

- **arXiv preprint:** Coming soon.
- **Code:** [Available on GitHub](https://github.com/CharlesJW222/VisualMontage).
- **Dataset:** Coming soon.

## Project page

The site is static HTML, CSS, and JavaScript. No build step, package installation, external fonts, analytics, or backend is required. Research figures are optimized WebP exports of the manuscript figures. The page includes model filters, an expandable numerical results table, keyboard-accessible finding tabs, and a figure viewer.

To preview locally from the repository root:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory website
```

Visit `http://127.0.0.1:8000/`. All asset links are relative, including for GitHub Pages deployment under `/VisualMontage/`.

### Update content

The following paths are relative to `website/`:

- `index.html`: title, author, affiliation, release news, research narrative, and figure captions.
- `assets/js/data.js`: reported model results, benchmark domain counts, and the sequence-length sweep. This is aggregate result data, not the benchmark release.
- `assets/css/style.css`: responsive layout and visual design.
- `assets/js/site.js`: result rendering and accessible interactions.
- `assets/images/`: original research figures exported for the web, favicon, and link-preview image.

When release links become available, replace their **Coming soon** labels and update the News section. Preserve the distinction between ASR, independently valid ASR, and end-to-end valid success. Source-order and random-order controls resample source frames; they do not isolate ordering alone. Reported audits and posting-naturalness evaluations are model-based.

### GitHub Pages

The workflow in [`../.github/workflows/deploy-website.yml`](../.github/workflows/deploy-website.yml) publishes the contents of `website/` to the existing project-page URL. It runs when website files or the workflow change on `main`, and can also be run manually from the Actions tab.

In **Settings → Pages → Build and deployment**, set **Source** to **GitHub Actions**, following the [GitHub Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages). The deployment artifact contains only `website/`, so future code and datasets elsewhere in the repository are not included in the website deployment.
