# Governance

MindIE SD follows a maintainer-led governance model for code review, protected-branch merging, and release readiness.

## Roles

### Branch keepers

Branch keepers are responsible for:

- release readiness checks
- protected-branch merge policy
- final merge authorization on protected branches

### Approvers

Approvers are responsible for:

- reviewing changes that are ready to merge
- validating technical quality, test evidence, and documentation impact
- confirming that user-visible changes are reflected in docs and release notes when needed

### Reviewers

Reviewers are responsible for:

- technical feedback
- reproduction assistance
- code, test, and documentation review

## Decision process

### Routine pull requests

For normal pull requests:

1. A clear issue, problem statement, or approved RFC should exist.
2. The contributor prepares code, tests, and documentation updates.
3. A reviewer provides technical review.
4. An approver confirms readiness.
5. A branch keeper or authorized maintainer merges to the protected branch.

### Release and governance decisions

For release and governance decisions:

1. Branch keepers coordinate release readiness and branch policy.
2. Approvers confirm code, documentation, and changelog state.
3. Release execution happens only on approved Ascend/NPU runners.

## Source of truth

Role membership is maintained in the repository:

- [`OWNERS`](../../../OWNERS) is the role source used by the repository workflow.

## Community standards

- Follow [`CODE_OF_CONDUCT.md`](../../../CODE_OF_CONDUCT.md).
- Keep tests, changelog entries, and documentation aligned with user-visible changes.
- Use an RFC before landing major interface, behavior, or release-policy changes.
