# Change Log

| Date | Change | Author |
|---|---|---|
| 2026-07-26 | Initial E7-F7/T7 plan drafted with seven issue-sized phases for fixed-shape map validation, volume evolution, conservative gas communication, fixed-capacity particle transport, resident scheduler integration, parity/conservation evidence, and final documentation. Preserved issue #1451 scope, E7-F4/E7-F5/E7-F6 dependencies, classifier diagnostics `none`, and explicit CFD/graph-capture/autodiff exclusions. | plan-feature-drafter |
| 2026-07-31 | Shipped E7-F7-P1 for issue #1507: `particula.execution.communication` supplies immutable direct-import-only declarations and deterministic read-only Warp payload validation, with co-located contract tests. No writer, transport, mutation, allocation, resource registration, hidden transfer/fallback, or package/top-level export was added; P2--P7 remain deferred. | plan-update-full |
