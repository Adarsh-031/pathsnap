# Member 4 integration and demo checklist

## Team handoffs

- [ ] **Member 1:** Share the supported trip-reader import, function signature, and return types for point, node, arc, and reference route data.
- [ ] **Member 3:** Share the OpenAPI contract or real redacted examples for `GET /health`, `GET /trips`, and `POST /predict`, including status codes/errors, coordinate order and path breaks, metric denominators, and timing units.
- [ ] **Member 3:** Provide the FastAPI startup command and artifact configuration, confirm access to all ten reserved trip IDs on Laptop 1, and confirm that scoring references never enter the prediction pipeline or API payloads.
- [ ] **Team:** Confirm the demo network, Internet availability for the map, and the checklist for the pre-lunch evaluation.

## Rehearsal and final demonstration

1. Generate the real training map from `00000001`; toggle the three layers and check that the map fits the trip. Keep this map separate from evaluator results.
2. Rehearse with permitted student-development trips first. Start the real API on Laptop 1 and leave its access logs visible.
3. On Laptop 2, verify LAN access and the health endpoint, then start the UI using the confirmed backend URL. Simulate an unreachable backend and a backend error during rehearsal; ensure the UI shows the error without reusing old results.
4. Confirm the final UI lists and accepts any of the ten reserved trip IDs during the authorized evaluation, not during development. Verify the selected trip reaches Laptop 1 and that the UI displays the returned prediction, accuracy, and timing. Keep evaluator reference labels and geometry server-side.
5. Capture a representative Folium map, the working UI, and a successful API request/response on Laptop 1's terminal. Do not capture or include evaluator reference files, private credentials, or fabricated results.
6. Before committing, inspect `git status` and the diff; stage only Member 4 files. Push the feature branch and open a PR into `master` without merging.

## Evidence for the consolidated report

Keep local HTML exports and screenshots outside the checkout (for example, in the sibling `PathSnap-artifacts` directory). The generation code belongs in the repository.

| Report section | Evidence to collect |
| --- | --- |
| Dataset understanding and EDA | Training trip `00000001` map screenshot with all three layers and legend; identify the split. Ask Member 1 for preprocessing, data checks, and feature rationale. |
| Model experiments and Bayesian tuning | Ask Member 2 for actual model comparisons, CV and student-test metrics, plots, and tuned-vs-untuned results. |
| HMM and Viterbi | Ask Member 3 for the classifier-only vs. decoded comparison, gap handling, and continuity results. |
| Folium visualization | Include a representative exported view in the PDF and reproducible generation code in GitHub. Add an actual prediction view when available. |
| API, UI, and deployment | Capture the real evaluator interface and a successful Laptop 1 terminal response. Explain which laptop runs each component and how requests travel between them. |

Each member provides the evidence and explanation for their own work. Test fixtures do not belong in the report as experimental results.
