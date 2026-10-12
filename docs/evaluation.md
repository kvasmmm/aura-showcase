# Testing & Technical Evaluation

> **Evaluation Audit:** Independent code and test verification performed on server environment.

---

## 1. Automated Test Suites (Independently Re-Run)

Test suites across both repositories were executed in isolated test environments:

| Repository / Module | Git Commit SHA | Environment | Command / Framework | Collected | Passed | Failed / Errors | Execution Time |
|---|---|---|---|---|---|---|---|
| **Aura Backend** | `f713a18` | Linux, Python 3.12.14, SQLite in-memory | `pytest app/tests -q` | 94 | **94** | 0 | 1.70s |
| **Aura Face Profiler** | `593f4e2` | Linux, Python 3.12.14 | `pytest tests/ -q` | 123 | **123** | 0 | 15.41s |
| **Frontend** | `f713a18` | Node.js, Vitest (jsdom) | Vitest | — | Unit tests present | 0 | — |

> **Audit Note on Aggregation:** Test counts belong to separate repositories with independent dependencies. They are reported individually and should not be combined. Backend tests run against in-memory SQLite rather than live production PostgreSQL.

### Key Subsystem Test Results:
- **Factorization Machine (`test_factorization_machine.py`, 7 tests):** Verified that sparse $O(k \cdot \text{nnz})$ predictions match naive $O(k \cdot \text{nnz}^2)$ reference implementations; verified analytical SGD gradients against finite-difference numerical approximations; verified pairwise ranking mode and warm-start `partial_fit` retention.
- **FM Interactions & Combos (`test_fm_interactions.py`, 8 tests):** Verified signature combo extraction, best-match integration before stability thresholds, and non-linear score dominance over linear baselines.
- **Active Learning (`test_active_learning.py`, 7 tests):** Verified score normalization, multi-strategy voting combinations, unseen item exclusion constraints, and strategy mathematical properties.
- **AL Recommendation Integration (`test_active_learning_recommendation_integration.py`, 2 tests):** Verified that active learning respects confirmed negative region exclusions and candidate filtering.

---

## 2. What Is NOT Evaluated (Honest Assessment)

To maintain strict technical integrity, the following limitations are explicitly noted:

1. **No Held-Out Recommendation Benchmarking:** Aura does not currently evaluate ranking performance using traditional offline recommendation metrics (e.g., Leave-One-Out Hit Rate, NDCG@K, MRR) against historical interaction logs.
2. **No Empirical Validation of FM over Linear Baseline:** While the FM model is mathematically verified to learn non-linear feature pairings, no statistical study has been performed comparing user satisfaction between pure linear EMA dot products and FM-augmented scoring on live traffic.
3. **No Filter-Bubble or Diversity Quantification:** Although active learning and heuristic exploration buckets are designed to promote catalog diversity, Gini coefficients and intra-list diversity metrics have not been formally tracked.
4. **No Ground-Truth Computer Vision Benchmark:** The Face Profiler quality-gate pass rate (85.88%) measures adherence to pose and bounding box thresholds, not metric landmark localization precision against annotated ground-truth datasets (e.g., 300-W or AFLW).
