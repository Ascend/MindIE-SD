# Community Governance

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-05T08:13:43.293Z pushedAt=2026-06-09T06:23:43.200Z -->

MindIE SD adopts a maintainer-led governance model for unified management of code review, protected branch merging, and version release.

## Role Definitions

### Branch keepers

Branch keepers are responsible for:

- Checking release readiness

- Enforcing merge strategy for protected branches

- Authorizing final merge on protected branches

### Approvers

Approvers are responsible for:

- Reviewing changes that are ready to merge

- Validating technical quality, test evidence, and documentation impact

- Ensuring user-facing changes are reflected in docs and release notes

### Reviewers

Reviewers are responsible for:

- Providing technical feedback

- Assisting with issue reproduction

- Reviewing code, tests, and documentation

## Decision-Making Process

### Daily Pull Requests (PRs)

Regular PRs are processed according to the following procedure:

1. A clear problem description, Issue, or approved RFC must exist first.

2. The contributor submits code, tests, and documentation updates.

3. Reviewers provide technical review comments.

4. Approvers confirm that the changes are ready for merging.

5. The Branch keeper or an authorized maintainer merges the change into the protected branch.

### Release and Governance Decisions

Release and governance matters are handled according to the following process:

1. Branch keepers coordinate release preparation and branch strategies.

2. Approvers verify code, documentation, and changelog status.

3. Releases are executed only on approved Ascend/NPU runners.

## Information Sources

The role list and the role sources used in repository workflows are based on [`OWNERS`](../../../OWNERS).

## Community Norms

- Comply with [`CODE_OF_CONDUCT.md`](../../../CODE_OF_CONDUCT.md).

- User-visible changes must be synchronized with updates to tests, documentation, and changelogs.

- Major interface changes, behavior changes, and release strategy changes should first be submitted as an RFC.
