# /start-phase

Read CLAUDE.md Tier 1 to get the current phase number and name.
Read .claude/skills/phase-context.md to get the FULL definition of the
current phase: deliverables, gate requirements, and out-of-scope items.

Output:
1. Current phase name and number
2. Exact deliverables for this phase (from phase-context.md)
3. Explicit out-of-scope items (anything not listed as a deliverable)
4. Gate requirements that must pass before commit

Then: "Confirm scope and I will begin. Type: confirmed, proceed"

Do NOT begin implementation until I send "confirmed, proceed".
