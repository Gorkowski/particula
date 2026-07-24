# Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-07-21 | Initial T6 feature plan drafted with E6-F5 dependency, default resampling, optional volume scaling, resampling-first precedence, fail-closed planning, and conservation requirements. | plan-feature-drafter |
| 2026-07-24 | Issue #1422 delivered E6-F6-P1's concrete CPU-only read-only planner and focused tests: strict frozen controls/sidecars/plans, all-box validation before resolution, deferred resampling-first policy selection, and float64 tuple-backed inventories. Commits, selection/resampling, scaling feasibility, discovery, exports, and GPU work remain deferred. | plan-update-full |
| 2026-07-24 | Issue #1423 delivered E6-F6-P2's CPU-only deterministic fixed-capacity equal-weight resampling reference: immutable detached cached-state plans, stable interval-sweep remaps, conservation/moment validation, and all-box-preflighted atomic commit. Co-located tests cover deterministic ordering, independent conservation, slot clearing, and invalid-plan no-mutation paths. Scaling, exports, discovery, and GPU work remain deferred. | plan-update-full |
