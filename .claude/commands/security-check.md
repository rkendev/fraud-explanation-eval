# /security-check

Act as Production Security Engineer. Review the current implementation for:

1. INJECTION SURFACE
   - Every agent that processes text fields: is sanitize_external_text() called?
   - Are the fields DeviceInfo, P_emaildomain, R_emaildomain sanitized BEFORE
     being passed to any LLM context?

2. SECRET HYGIENE
   - Run: grep -r "api_key\s*=\s*['\"]" src/ --include="*.py"
   - Run: grep -r "sk-\|anthropic" src/ --include="*.py"
   - Either should return nothing. Any match = RED.

3. AUDIT TRAIL
   - Is cost_log.jsonl being appended for every LLM call?
   - Does each record have transaction_id, agent_name, timestamp?

4. API AUTHENTICATION (Phase 6+)
   - Is X-API-Key middleware present on all FastAPI routes?
   - Is rate limiting applied?

5. CUSTOMER DATA ISOLATION
   - Is fraud_probability accessible in customer-mode API responses?
   - If yes: RED — must be stripped at the API layer.

Output: ✅/❌ per item with line references.
