# /agent-review

Conduct a formal phase gate review as four simultaneous roles:
- Senior AI Systems Architect: spec compliance, failure mode coverage, state machine correctness
- Production Security Engineer: injection surface, secret handling, audit logging
- Staff QA Engineer: coverage %, hallucination tests, adversarial tests, golden scenarios
- Engineering Manager: cost tracking wired, no zero-cost fields, budget within COST_BUDGET.md

## For each role, check:

### Architect
- [ ] Implementation matches AGENT_SPEC.md for this phase (input contract, output contract)
- [ ] All documented failure modes have a corresponding test
- [ ] confidence_tier boundaries tested at exact thresholds
- [ ] LangGraph state keys have explicit reducers (no last-write-wins)

### Security
- [ ] Any agent consuming text fields: sanitize_external_text() called?
- [ ] No secrets in source code (check .env usage)
- [ ] Audit log (cost_log.jsonl) appended after every LLM call
- [ ] New API surfaces: auth middleware in place?

### QA
- [ ] Branch coverage ≥ 85% on all new modules (run: make coverage)
- [ ] All required test categories from AGENT_SPEC.md present
- [ ] Hallucination boundary tests: at least one test per ExplanationResult invariant
- [ ] Golden scenarios: relevant scenarios for this phase pass

### EM
- [ ] token_cost_usd > 0.0 on all LLM-producing schemas
- [ ] record_agent_call() called in every agent's return path
- [ ] cost_log.jsonl has entries from this phase's test runs
- [ ] make cost-report shows no unresolved budget breaches

## Verdict per role: ✅ GREEN | ⚠️ AMBER: [description] | ❌ RED: [description]

## Final gate: GREEN | AMBER | RED

## If GREEN or AMBER (documented exceptions only):
Output exactly:
"Gate passed. Next steps:
1. make test (confirm green)
2. make cost-report (confirm no breaches)
3. git commit -m 'feat(phase-N): [name] — [X] tests, [Y]% coverage'
4. git push
5. make advance-phase PHASE='[next phase name]'
6. /clear
7. /start-phase"

## If RED:
Output: "Gate BLOCKED. Resolve before proceeding:"
List every RED item. Do not suggest committing.
