# SkillCortex Router V1 Design

Promote `alternating_skill` without changing the validated protected router.

`SkillCortexRouterV1` delegates to `ProtectedSkillRouterWithoutFailureBorn`
except for the two explicit `alternating` routes. The existing quarantined
candidate router and experiment artifacts remain unchanged.

The integration report is a deterministic rename/projection of the completed
failure-born experiment summary. It performs no training or inference and
records that fact in its output.

Tests cover the semantic gate, baseline delegation, promoted schema/config,
unchanged benchmark and historical artifacts, and both report sections.
