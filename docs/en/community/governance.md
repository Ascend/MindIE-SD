# Community Governance

MindIE SD adopts a maintainer-led governance model for code review, protected branch merging, and release management.

## Role Definitions

### Branch Keepers

Branch keepers are responsible for:

- Release readiness checks
- Protected branch merge policies
- Final merge authorization on protected branches

### Approvers

Approvers are responsible for:

- Reviewing changes that are ready for merging
- Verifying technical quality, test evidence, and documentation impact
- Confirming that user-visible changes are synced to documentation and release records

### Reviewers

Reviewers are responsible for:

- Providing technical review feedback
- Assisting with issue reproduction
- Reviewing code, tests, and documentation

## Decision Process

### Routine Pull Requests

Regular PRs follow this process:

1. Clear problem description, Issue, or approved RFC comes first.
2. Contributor submits code, tests, and documentation updates.
3. Reviewers provide technical review feedback.
4. Approvers confirm that the changes are ready for merging.
5. Branch keepers or authorized maintainers merge changes into protected branches.

### Release and Governance Decisions

Release and governance matters follow this process:

1. Branch keepers coordinate version release preparation and branch strategy.
2. Approvers confirm code, documentation, and changelog status.
3. Releases are only executed on approved Ascend/NPU runners.

## Information Sources

The role roster and repository workflow role sources are based on [`OWNERS`](../../../OWNERS).

## Community Guidelines

- Comply with [`CODE_OF_CONDUCT.md`](../../../CODE_OF_CONDUCT.md).
- User-visible changes must sync test, documentation, and changelog updates.
- Major API changes, behavioral changes, and release policy changes should submit an RFC first.
