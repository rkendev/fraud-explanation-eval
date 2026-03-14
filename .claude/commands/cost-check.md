# /cost-check

Run cost analysis for this phase:

1. Run: make cost-report
2. Report: total spend, avg per transaction, budget breach count
3. Check COST_BUDGET.md: are all agent calls within per-agent budget?
4. Check: is the cheap/strong tier split reasonable (cheap > 75%)?
5. Flag any agent call with token_cost_usd == 0.0 in cost_log.jsonl
   (this means record_agent_call() was not called or cost was not propagated)

If any budget breaches: list transaction_id and agent_name.
If any zero-cost records: list them as RED items.
