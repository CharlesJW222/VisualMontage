# VisualMontage

**Manipulating the Beliefs of Vision-Language Agents with Authentic Image Sequences**

**Jinwei Hu<sup>1,*</sup>, Wei Lai<sup>1,*</sup>, Zhuoyun Li<sup>1</sup>, Youcheng Sun<sup>2</sup>, Yi Dong<sup>1</sup>, Xiaowei Huang<sup>1</sup>**

<sup>1</sup> School of Computer Science and Informatics, University of Liverpool, UK

<sup>2</sup> Department of Computer Science, MBZUAI, United Arab Emirates

<sup>*</sup> Equal contribution.

[Project page](https://CharlesJW222.github.io/VisualMontage/)

VisualMontage studies visual cognitive collusion: authentic images and locally faithful captions can jointly induce a conclusion contradicted by the complete source context. This repository currently contains the research project page.

## News

- **arXiv preprint:** Coming soon.
- **Code:** Coming soon.
- **Dataset:** Coming soon.

## Project page

The site is static HTML, CSS, and JavaScript. No build step, package installation, external fonts, analytics, or backend is required. Research figures are optimized WebP exports of the manuscript figures. The page includes model filters, an expandable numerical results table, keyboard-accessible finding tabs, and a figure viewer.

To preview locally:

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Visit `http://127.0.0.1:8000/`. All asset links are relative, including for GitHub Pages deployment under `/VisualMontage/`.

### Update content

- `index.html`: title, author, affiliation, release news, research narrative, and figure captions.
- `assets/js/data.js`: reported model results, benchmark domain counts, and the sequence-length sweep. This is aggregate result data, not the benchmark release.
- `assets/css/style.css`: responsive layout and visual design.
- `assets/js/site.js`: result rendering and accessible interactions.
- `assets/images/`: original research figures exported for the web, favicon, and link-preview image.

When release links become available, replace their **Coming soon** labels and update the News section. Preserve the distinction between ASR, independently valid ASR, and end-to-end valid success. Source-order and random-order controls resample source frames; they do not isolate ordering alone. Reported audits and posting-naturalness evaluations are model-based.

### GitHub Pages

Publish from the **main** branch and **/ (root)** folder in **Settings → Pages**. The `.nojekyll` file enables a plain static site. Publishing the page does not require uploading the manuscript PDF or research data.

The page layout takes inspiration from [SIDA](https://hzlsaber.github.io/projects/SIDA/); the HTML, styles, and interactions here are implemented independently.
