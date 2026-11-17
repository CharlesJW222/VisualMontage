"use strict";

(() => {
  const data = window.VISUALMONTAGE_DATA;
  const number = value => value.toFixed(2);
  const groupName = group => group === "proprietary" ? "Proprietary" : "Open-weight";

  // Only trusted, locally maintained research data is rendered here.
  const chart = document.getElementById("model-chart");
  chart.innerHTML = `<div class="chart-scale" aria-hidden="true"><span>Model</span><div class="scale-values"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div></div>`;
  data.models.forEach(model => {
    const row = document.createElement("div");
    row.className = "model-row";
    row.dataset.group = model.group;
    row.innerHTML = `<div class="model-label">${model.name}<small>${groupName(model.group)}</small></div>
      <div class="bar-region"><div class="bar-pair" role="img" aria-label="Source-order ${number(model.source)} percent; VisualMontage ${number(model.asr)} percent">
        <div class="bar-track" style="--value:${model.source}%" aria-hidden="true"><div class="bar-fill"></div><span class="bar-value">${number(model.source)}</span></div>
        <div class="bar-track montage" style="--value:${model.asr}%" aria-hidden="true"><div class="bar-fill"></div><span class="bar-value">${number(model.asr)}</span></div>
      </div></div>`;
    chart.append(row);
  });

  const table = document.getElementById("model-table");
  ["proprietary", "open"].forEach(group => {
    const rows = [...data.models.filter(model => model.group === group), data.means.find(model => model.group === group)];
    rows.forEach((model, index) => {
      const row = document.createElement("tr");
      if (index === rows.length - 1) row.className = "mean-row";
      row.innerHTML = `<th scope="row">${model.name}</th>` +
        [model.noEvidence, model.source, model.random, model.asr, model.hc, model.valid, model.confidence].map(value => `<td>${number(value)}</td>`).join("");
      table.append(row);
    });
  });

  document.querySelectorAll("[data-model-filter]").forEach(button => {
    button.addEventListener("click", () => {
      const group = button.dataset.modelFilter;
      document.querySelectorAll("[data-model-filter]").forEach(other => {
        const active = other === button;
        other.classList.toggle("is-active", active);
        other.setAttribute("aria-pressed", String(active));
      });
      let count = 0;
      chart.querySelectorAll(".model-row").forEach(row => {
        row.hidden = group !== "all" && row.dataset.group !== group;
        if (!row.hidden) count++;
      });
      document.getElementById("filter-status").textContent = group === "all" ? `Showing all ${count} models.` : `Showing ${count} ${groupName(group).toLowerCase()} models.`;
    });
  });

  const domainChart = document.getElementById("domain-chart");
  const maxCases = Math.max(...data.domains.map(domain => domain.cases));
  data.domains.forEach(domain => {
    const row = document.createElement("div");
    row.className = "domain-row";
    row.innerHTML = `<div class="domain-row-label"><span>${domain.name}</span><strong>${domain.cases}</strong></div><div class="domain-track" aria-hidden="true"><span style="--value:${domain.cases / maxCases * 100}%"></span></div>`;
    domainChart.append(row);
  });

  const lengthChart = document.getElementById("length-chart");
  data.lengths.forEach(result => {
    const column = document.createElement("div");
    column.className = "length-column";
    column.setAttribute("role", "img");
    column.setAttribute("aria-label", `${result.items} items: ASR ${number(result.asr)} percent, end-to-end valid success ${number(result.e2e)} percent`);
    column.innerHTML = `<div class="length-bar" style="--height:${result.asr}%" aria-hidden="true"><span class="length-value">${result.asr.toFixed(1)}</span></div><div class="length-bar is-valid" style="--height:${result.e2e}%" aria-hidden="true"><span class="length-value">${result.e2e.toFixed(1)}</span></div><span class="length-label" aria-hidden="true">${result.items}</span>`;
    lengthChart.append(column);
  });

  // Keyboard support for the WAI-ARIA tab pattern.
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  function selectTab(tab, focus = false) {
    tabs.forEach(other => {
      const selected = other === tab;
      other.setAttribute("aria-selected", String(selected));
      other.tabIndex = selected ? 0 : -1;
      document.getElementById(other.getAttribute("aria-controls")).hidden = !selected;
    });
    if (focus) tab.focus();
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => selectTab(tab));
    tab.addEventListener("keydown", event => {
      let next;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = tabs.length - 1;
      if (next !== undefined) {
        event.preventDefault();
        selectTab(tabs[next], true);
      }
    });
  });

  const navToggle = document.querySelector(".nav-toggle");
  const navLinks = document.getElementById("nav-links");
  function closeMenu() {
    navLinks.classList.remove("is-open");
    navToggle.setAttribute("aria-expanded", "false");
    navToggle.setAttribute("aria-label", "Open navigation");
  }
  navToggle.addEventListener("click", () => {
    const open = !navLinks.classList.contains("is-open");
    navLinks.classList.toggle("is-open", open);
    navToggle.setAttribute("aria-expanded", String(open));
    navToggle.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
  });
  navLinks.querySelectorAll("a").forEach(link => link.addEventListener("click", closeMenu));
  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && navLinks.classList.contains("is-open")) {
      closeMenu();
      navToggle.focus();
    }
  });
  document.addEventListener("click", event => {
    if (!event.target.closest(".nav")) closeMenu();
  });
  window.matchMedia("(min-width: 761px)").addEventListener("change", closeMenu);

  // Without native dialog support, figure links still open the original image.
  const dialog = document.getElementById("figure-dialog");
  if (typeof dialog.showModal === "function") {
    const enlarged = document.getElementById("lightbox-image");
    let opener = null;
    document.querySelectorAll("[data-lightbox]").forEach(link => {
      link.addEventListener("click", event => {
        if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
        event.preventDefault();
        opener = link;
        enlarged.src = link.href;
        enlarged.alt = link.querySelector("img").alt;
        document.getElementById("lightbox-caption").textContent = link.dataset.caption;
        document.getElementById("lightbox-original").href = link.href;
        document.body.classList.add("modal-open");
        dialog.showModal();
      });
    });
    dialog.querySelector(".lightbox-close").addEventListener("click", () => dialog.close());
    dialog.addEventListener("click", event => {
      if (event.target !== dialog) return;
      const bounds = dialog.getBoundingClientRect();
      if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
    });
    dialog.addEventListener("close", () => {
      document.body.classList.remove("modal-open");
      if (opener) opener.focus({ preventScroll: true });
    });
  }

  if ("IntersectionObserver" in window) {
    const links = [...navLinks.querySelectorAll("a")];
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        links.forEach(link => {
          const current = link.hash === `#${entry.target.id}`;
          link.classList.toggle("is-current", current);
          if (current) link.setAttribute("aria-current", "location");
          else link.removeAttribute("aria-current");
        });
      });
    }, { rootMargin: "-10% 0px -55% 0px", threshold: 0 });
    links.forEach(link => observer.observe(document.querySelector(link.hash)));
  }
})();
