/* Reported aggregate results, not the benchmark release.
 * Source: VisualMontage manuscript model, benchmark, and length tables.
 * Rates are percentages; confidence is on [0, 1].
 */
window.VISUALMONTAGE_DATA = {
  models: [
    { name: "GPT-6-sol", group: "proprietary", noEvidence: 0.37, source: 8.39, random: 7.21, asr: 62.28, hc: 44.30, valid: 57.61, confidence: 0.87 },
    { name: "GPT-6-luna", group: "proprietary", noEvidence: 0.40, source: 3.68, random: 3.88, asr: 66.14, hc: 40.18, valid: 60.40, confidence: 0.84 },
    { name: "GPT-5.6-terra", group: "proprietary", noEvidence: 0.11, source: 11.23, random: 11.84, asr: 60.44, hc: 53.53, valid: 55.44, confidence: 0.91 },
    { name: "GPT-5.6-luna", group: "proprietary", noEvidence: 0.81, source: 14.60, random: 15.42, asr: 70.37, hc: 62.72, valid: 64.74, confidence: 0.90 },
    { name: "Claude Haiku 5.5", group: "proprietary", noEvidence: 0.55, source: 6.66, random: 7.06, asr: 48.57, hc: 17.61, valid: 44.60, confidence: 0.72 },
    { name: "Claude Sonnet 5.5", group: "proprietary", noEvidence: 0.48, source: 1.10, random: 2.09, asr: 63.75, hc: 29.67, valid: 58.57, confidence: 0.76 },
    { name: "Qwen3-VL-4B-Inst", group: "open", noEvidence: 0.04, source: 11.25, random: 10.07, asr: 77.43, hc: 65.85, valid: 71.10, confidence: 0.89 },
    { name: "Qwen3-VL-8B-Inst", group: "open", noEvidence: 0.18, source: 11.91, random: 10.48, asr: 76.25, hc: 75.33, valid: 70.18, confidence: 0.95 },
    { name: "Qwen3-VL-32B-Inst", group: "open", noEvidence: 3.31, source: 7.35, random: 5.55, asr: 79.04, hc: 78.16, valid: 73.05, confidence: 0.90 },
    { name: "InternVL3.5-8B", group: "open", noEvidence: 0.26, source: 2.76, random: 3.29, asr: 63.86, hc: 51.47, valid: 58.93, confidence: 0.84 },
    { name: "InternVL3.5-14B", group: "open", noEvidence: 3.35, source: 18.92, random: 19.31, asr: 47.68, hc: 46.29, valid: 43.90, confidence: 0.89 },
    { name: "InternVL3.5-38B", group: "open", noEvidence: 4.19, source: 1.87, random: 2.04, asr: 67.98, hc: 67.46, valid: 62.54, confidence: 0.89 }
  ],
  means: [
    { name: "Proprietary mean", group: "proprietary", noEvidence: 0.45, source: 7.61, random: 7.92, asr: 61.93, hc: 41.34, valid: 56.89, confidence: 0.83 },
    { name: "Open-weight mean", group: "open", noEvidence: 1.89, source: 9.01, random: 8.46, asr: 68.71, hc: 64.09, valid: 63.28, confidence: 0.89 }
  ],
  domains: [
    { name: "Emergency Response", cases: 101 },
    { name: "Search & Rescue", cases: 121 },
    { name: "Public Safety", cases: 65 },
    { name: "Aerospace Operations", cases: 38 },
    { name: "Industrial Operations", cases: 77 },
    { name: "Animal Rescue", cases: 62 },
    { name: "Sports Training", cases: 42 },
    { name: "Film & Animation", cases: 38 }
  ],
  lengths: [
    { items: 3, asr: 59.39, e2e: 25.93 },
    { items: 6, asr: 64.85, e2e: 33.90 },
    { items: 10, asr: 60.00, e2e: 20.97 },
    { items: 15, asr: 72.50, e2e: 9.86 },
    { items: 20, asr: 56.25, e2e: 7.35 },
    { items: 30, asr: 62.00, e2e: 5.56 }
  ]
};
