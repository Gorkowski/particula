# Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-09-02 | P3 (issue #1569) did not land: its E8-F4-P2 native-capture owner, opaque graph handle, captured prepared-plan owner, and `capture_launch()` runtime adapter were unavailable. No code, tests, or user documentation changed. | plan-update-full |
| 2026-09-02 | P1 (issue #1567) is directly blocked by the absent E8-F3 capture-resource carrier/contract. P2 (issue #1568) is directly blocked by absent P1; E8-F3 is transitive. No code, tests, or user documentation changed. | plan-update-full |
| 2026-08-30 | Aligned replay teardown with E8-F1: deterministic Python-owner retirement is required and native release is conditional on a documented, version-qualified public Warp API | user decision |
| 2026-08-30 | Initial E8-F4 plan drafted with five phases for prepared-plan capture, complete fixed-sequence recording, guarded replay, lifecycle invalidation, three-way validation, and documentation. Classifier diagnostics preserved as none. | plan-feature-drafter |
