# api_contract_fastapi_skill v2 Output Contract Design

## Status

- Skill: `api_contract_fastapi_skill`
- Scope: output-contract revision design only
- Current decision: v1 remains rejected
- Promotion status: do not promote
- Activation status: do not activate
- Registration status: do not register
- Routing status: do not route
- Isolation status: keep quarantined

This document defines a v2 output-contract revision for `api_contract_fastapi_skill` without retraining, without creating a new adapter, and without mutating any v1 data, holdout data, manifests, frozen benchmarks, router behavior, registry behavior, or `data/eval.jsonl`.

## 1. Why v1 Failed

The v1 adapter remains rejected. That decision is preserved and is not reopened by this document.

### Postmortem Evidence

- Primary recommendation: `revise_output_contract_before_retraining`
- Candidate train execution pass rate: `0%`
- Candidate holdout execution pass rate: `0%`
- Frozen FastAPI benchmark pass rate: `2.08%`
- Non-target regression versus router baseline: `32` pass-to-fail
- `alternating_skill` regression: `1` pass-to-fail
- `143/240` FastAPI generations hit the `256`-token cap
- `120/240` FastAPI generations were syntax-invalid
- `147/240` FastAPI generations had no route decorator
- `59/240` FastAPI generations lacked `app = FastAPI()`
- Fuzzy similarity improved, but executable correctness did not

### Failure Interpretation

v1 failed primarily as an output-shape and artifact-completeness problem, not as a pure semantic-nearest-neighbor problem. The evidence shows that the model frequently produced outputs that looked somewhat related to the target domain while failing to emit executable FastAPI artifacts.

The main failure modes were:

- truncation from an undersized generation budget;
- ambiguous target shape, likely causing mixed or incomplete file forms;
- invalid Python syntax at a high rate;
- omission of contract-critical anchors such as route decorators and `app = FastAPI()`;
- contamination outside the target task surface, demonstrated by non-target regressions.

### Preserved Decision

v1 remains rejected. It must not be promoted, activated, registered, routed to, or retrained from in its current form. All v1 frozen data and artifacts remain preserved as rejected evidence.

## 2. v2 Goal

The v2 goal is to revise only the output contract so that future retraining, if later authorized, has a better-defined and lower-entropy target.

### v2 Objectives

- reduce output-shape failure;
- reduce truncation risk;
- preserve FastAPI contract learning;
- avoid non-target contamination;
- keep the skill quarantined until explicit gates are met.

### Non-Goals

- no retraining in this phase;
- no new adapter creation in this phase;
- no v2 data generation in this phase;
- no router changes;
- no registry changes;
- no benchmark edits;
- no mutation of `data/eval.jsonl`;
- no mutation or deletion of v1 artifacts.

## 3. Output Contract Alternatives

This section evaluates four candidate output contracts for v2.

### Option A: Full-File Generation with Higher Generation Cap

#### Description

Each row expects one complete Python file as output, typically a minimal FastAPI app or a complete test file. The generation cap is increased enough to avoid routine truncation.

#### Benefits

- simple training target shape;
- simple evaluation fixture because the output is directly executable as a file;
- preserves end-to-end FastAPI contract learning;
- avoids reconstruction complexity.

#### Risks

- still vulnerable to long-output drift;
- still susceptible to file-shape ambiguity if prompt/task boundaries are weak;
- higher cap can hide contract inefficiency rather than fix it;
- complete-file generation can still mix app and tests unless explicitly forbidden.

#### Assessment

Better than v1 only if target size is aggressively constrained. Raising the cap alone does not solve ambiguous artifact shape.

### Option B: Smaller Single-File Target

#### Description

Each row expects exactly one small, self-contained file with a narrow scope. The file must fit under a strict length budget and contain one unambiguous artifact type: app file, test file, or refactor target, depending on task type.

#### Benefits

- directly attacks truncation by reducing target size;
- minimizes file-shape ambiguity;
- keeps evaluation simple because outputs remain executable files;
- preserves core FastAPI contract learning in a complete artifact;
- reduces opportunity for app-plus-test mixing.

#### Risks

- requires careful curation so examples remain representative;
- may reduce coverage of larger multi-endpoint patterns;
- needs strict gating to exclude oversized references.

#### Assessment

Strong candidate because it reduces the two biggest observed failure modes: truncation and ambiguous shape.

### Option C: Patch/Diff-Style Output

#### Description

Each row expects a patch against a provided base file rather than a full file. The harness reconstructs the final executable artifact by applying the patch before execution.

#### Benefits

- potentially much smaller outputs;
- reduces redundant boilerplate emission;
- may improve debugging and refactor tasks where a base file already exists.

#### Risks

- adds reconstruction complexity in the harness;
- creates a second failure mode where patch syntax is invalid even if the intended code is correct;
- partial edits can hide missing global structure such as missing imports or app initialization;
- poor fit for pure generation tasks that need a complete app from nothing.

#### Assessment

Useful for debugging or refactor-only settings, but risky as the primary universal contract because patch syntax and reconstruction become new sources of failure.

### Option D: Subskill Decomposition

#### Description

Split the task surface into narrower output contracts, such as route-only creation, schema-only creation, tests-only creation, or bug-fix-only tasks.

#### Benefits

- lowers per-task entropy;
- can isolate different failure classes;
- allows smaller targets and more precise fixtures.

#### Risks

- increases dataset and evaluation design complexity;
- risks teaching fragments that no longer reconstruct into a valid FastAPI app contract;
- can accidentally create hidden dependencies between subtasks;
- may require new orchestration assumptions that are too close to adapter redesign.

#### Assessment

Promising as a later refinement, but too complex as the first corrective move. It risks shifting the problem from output shape to cross-subtask assembly.

## 4. Recommended v2 Contract

### Primary Recommendation

Adopt **smaller single-file target** as the primary v2 output format.

### Why This Is Recommended

This format best minimizes truncation and file-shape ambiguity while preserving executable FastAPI contract learning.

It directly addresses the observed v1 failures:

- outputs stay short enough to fit well within the active generation cap;
- each row has exactly one expected artifact shape;
- fixtures can execute the file directly with minimal reconstruction logic;
- app-critical anchors such as imports, `app = FastAPI()`, route decorators, and response behavior remain visible in one coherent artifact;
- non-target contamination risk is reduced because examples become narrower and less likely to induce mixed-mode outputs.

### Secondary Allowance

Patch-style output may be considered later for debugging or refactor subsets, but it is not the primary recommended v2 contract in this design.

## 5. v2 Task Shapes

All v2 tasks must emit exactly one unambiguous artifact per row.

### `fastapi_contract_generation`

- Output shape: one complete Python app file
- Allowed content: imports, `app = FastAPI()`, one or more route handlers, minimal supporting models or helpers required by that file
- Disallowed content: tests, shell commands, prose, multiple files, patch syntax
- Target bias: keep to minimal single-endpoint or small multi-endpoint apps that fit comfortably within budget

### `fastapi_contract_debugging`

- Output shape: one complete corrected Python app file
- Allowed content: a full repaired app file corresponding to the provided buggy source task
- Disallowed content: partial snippets, explanation text, tests, mixed patch-plus-file output unless the task family is explicitly redefined as patch-based
- Target bias: fixture should verify that the repaired app executes and satisfies the contract

### `fastapi_contract_test_generation`

- Output shape: one complete Python test file
- Allowed content: tests only, including imports and test client setup required for execution against the provided app fixture
- Disallowed content: application source unless the task explicitly defines that the app is embedded, which v2 should avoid
- Target bias: tests must target a provided app fixture, not generate the app and tests together

### `fastapi_contract_refactor`

- Output shape: one complete refactored Python app file
- Allowed content: a full rewritten app file preserving specified behavior while improving structure or clarity
- Disallowed content: prose, tests, patches, multiple alternative outputs
- Target bias: behavior-preserving refactors only; fixtures must assert semantic equivalence for the intended contract

## 6. v2 Artifact Size Budget

### Maximum Target Length

- Maximum reference target length: `160` non-empty lines
- Preferred target length: `<= 100` non-empty lines
- Hard rejection threshold: any reference target above `160` non-empty lines is excluded from v2

### Maximum Expected Generation Tokens

- Active expected generation budget: `<= 220` tokens for the reference target median family
- Comfort requirement: every reference target must fit comfortably within the active generation cap, with at least `25%` headroom relative to the cap used for the task family

### Rejection Gate

Reject any proposed v2 target row if any of the following are true:

- expected artifact length exceeds the line budget;
- expected artifact routinely approaches the active cap without headroom;
- the row requires boilerplate or fixture detail that pushes the output toward truncation;
- the output cannot be expressed as one unambiguous artifact.

### Practical Constraint

The exact active generation cap may be tuned later, but v2 references must be curated so they are well below that cap rather than merely squeezing under it.

## 7. v2 File-Shape Rules

Each v2 row must define exactly one expected artifact shape.

### Universal Rules

- one row must map to one artifact;
- one artifact must map to one file;
- one file must have one declared purpose.

### Allowed Artifact Types by Task

- generation: complete app file only;
- debugging: complete corrected app file only;
- test generation: complete test file only;
- refactor: complete refactored app file only.

### Forbidden Shapes

- mixed app-and-test outputs unless the task explicitly requires tests;
- partial snippets unless the task family is explicitly patch-based;
- multiple files in one target;
- prose before or after code;
- placeholder ellipses;
- incomplete imports or omitted top-level scaffolding that the fixture expects.

### Required App Anchors for App-File Tasks

For generation, debugging, and refactor tasks, the expected file must include all required top-level anchors for executability when the task semantics demand them, including:

- valid Python syntax;
- required imports;
- `app = FastAPI()`;
- at least one route decorator;
- route handler definitions required by the prompt/fixture.

### Required Test Anchors for Test Tasks

For test-generation tasks, the expected file must include:

- valid Python syntax;
- the testing framework imports used by the fixture;
- imports of the provided app fixture or module-under-test;
- executable tests with deterministic assertions.

## 8. v2 Evaluation Compatibility

### Output Execution Model

The recommended v2 contract uses direct executable files.

- generation outputs are executed as full app files;
- debugging outputs are executed as full repaired app files;
- test-generation outputs are executed as full test files against provided app fixtures;
- refactor outputs are executed as full app files and checked for preserved behavior.

### Fixture Compatibility

Because the primary recommendation is full single-file output, fixtures should not need patch application or block reconstruction for the main v2 path.

### If Patch/Block Tasks Are Added Later

If a later non-primary task family uses patches or structured code blocks, the harness must:

- start from a deterministic base artifact;
- reconstruct exactly one executable file;
- validate patch parseability before execution;
- reject outputs that cannot reconstruct into a single valid file.

That reconstruction path is explicitly out of scope for the primary v2 recommendation.

## 9. v2 Leakage and Independence

v2 must remain isolated from v1 and from frozen evaluation surfaces.

### Required Separation

- do not mutate or overwrite any v1 train data;
- do not mutate or overwrite any v1 holdout data;
- do not mutate or overwrite any v1 manifest;
- do not mutate or overwrite any v1 candidate artifacts;
- do not mutate the frozen FastAPI benchmark;
- do not mutate `data/eval.jsonl`.

### Independence Requirements

- v2 design assets must live in separate paths or naming scopes from v1 assets;
- train and holdout references must remain independent;
- no row may be copied from the frozen benchmark into training data;
- no benchmark-derived target may be transformed into a reference target for v2;
- candidate assets remain quarantined and non-routable.

## 10. v2 Success Gates Before Retraining

Retraining is not authorized by this document. Before any retraining request can even be considered, the following static and dry-run gates must pass.

### Required Pre-Retraining Gates

- no reference target exceeds the defined length budget;
- every row has exactly one expected output shape;
- every reference fixture passes on the curated reference output;
- every generated target is executable after any required reconstruction, if reconstruction is used;
- train and holdout remain independent;
- no benchmark leakage is present;
- candidate remains quarantined and inactive.

### Additional Recommended Dry-Run Checks

- schema check for task-type to artifact-type consistency;
- syntax parse pass for every Python target;
- task-family audit proving no mixed app-plus-test rows outside explicit test tasks;
- size-distribution audit demonstrating comfortable cap headroom.

## 11. v2 Retraining Gate

Retraining remains unauthorized after this design.

### Retraining May Only Be Authorized If

- the v2 contract is finalized in writing;
- all pre-retraining success gates pass;
- a quarantined v2 dataset is generated in a separate namespace from v1;
- reference fixtures prove the outputs are executable in their declared shape;
- an approval step explicitly authorizes retraining.

### Current State

None of those steps are authorized by this document. This document is design only.

## 12. Rejection Criteria

The v2 effort must be immediately rejected or halted if any of the following conditions occur.

### Immediate Rejection Criteria

- zero train execution pass rate;
- zero holdout execution pass rate;
- no improvement on the fixed frozen FastAPI benchmark;
- any `alternating_skill` pass-to-fail regression;
- unbounded non-target regression;
- any router mutation;
- any registry mutation;
- any activation or promotion of the quarantined candidate before explicit authorization.

### Additional Process Rejection Criteria

- mutation of v1 train, holdout, or manifest files;
- mutation of frozen benchmark assets;
- mutation of `data/eval.jsonl`;
- creation of a replacement adapter before design gates are passed;
- generation of ambiguous multi-artifact targets.

## Recommended Next Phase

The next recommended phase is **contract-spec validation only**.

That phase should:

- formalize the v2 artifact schema;
- dry-run the size and shape gates against candidate reference examples;
- prove fixture compatibility for each task family;
- keep the candidate quarantined;
- stop before any data generation or retraining.

## Decision Summary

- v1 rejection is preserved.
- v2 should revise the output contract before any retraining is considered.
- The recommended v2 output contract is **smaller single-file target**.
- The skill remains quarantined and unauthorized for promotion, activation, registration, routing, adapter creation, or retraining.