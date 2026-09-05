# Insurance Claims SOP Agent

A conversational insurance-claims support agent that follows a fixed,
auditable workflow (SOP) while still holding a natural conversation. The
stack is entirely local and free to run: Python/FastAPI backend, a local
**Ollama** model (`qwen3.5:4b` by default) for phrasing, and a single-page
browser test UI. No paid APIs or API keys are involved anywhere. If Ollama
isn't running, the agent still works end-to-end via a deterministic
template engine — only the wording gets a little less conversational.

```
VERIFY_ID -> RESOLVE_INTENT -> PROCESS_CASE -> POST_PROCESS
```

## Design

Every decision that matters for safety or correctness — whether the caller
is verified, which claim/intent applies, what facts may be disclosed, when
to escalate to a human — is made by plain deterministic Python in
`app/sop/*.py` and `app/nlu/*.py`. The LLM is only ever handed an
already-decided, already-fact-checked draft reply and asked to phrase it
more naturally (`app/llm/compose.py`); its output is then re-scanned and
discarded in favor of the safe draft if it introduces any fact, number, or
case ID that wasn't already there. This keeps the SOP itself immune to
prompt-level drift or small-model unreliability, while still allowing free
LLM reasoning exactly where the spec calls for it: interpreting messy
questions, picking the right claim from context, and casual phrasing.

## Features

- **VERIFY_ID** — strict identity gate. Requires 3-of-5 whitelisted PII
  fields (full name, DOB, phone, email, SSN/ID last-4) to match a single
  policyholder record before anything else is disclosed. Handles partial
  answers, refusals (empathy + an explanation of *why*, plus an offer of an
  alternate field), and a full representative/consent sub-flow: if a caller
  says they're calling on someone else's behalf, the agent checks
  `representatives.json`, then simulates requesting the policyholder's
  consent and polls a scenario from `consent_scenarios.json` turn by turn
  (`default` approves quickly; `timeout` never approves and escalates to a
  human) — all without disclosing account details in the meantime.
- **Cross-phase memory** — hints volunteered before verification (case
  type, status, month, an explicit case ID) are captured immediately and
  reused the moment the caller is verified, so nothing has to be repeated.
  This lets the agent go from "hello" to a fully grounded, resolved-case
  answer in a single turn when the caller volunteers enough detail up
  front.
- **RESOLVE_INTENT** — freer reasoning: filters the verified caller's
  claims by whatever hints/keywords are available, disambiguates when
  several claims could match, and classifies intent into `denial_question /
  status_inquiry / document_submission / next_steps /
  general_claim_question`.
- **PROCESS_CASE** — grounded Q&A assembled only from `claims.json` +
  `required_document_guideline.json` (denial reason, appeal deadline,
  per-document guidance and alternatives, and a keyword-matched
  follow-up-FAQ table). The LLM may phrase this content, but every fact in
  it is retrieved by code first.
- **POST_PROCESS** — offers an email summary (what was discussed, claim
  status/outcome, follow-up items), respects a yes/no either way, and can
  loop back into PROCESS_CASE if the caller has one more question before
  the call actually ends.
- **Scope guard** — declines out-of-scope questions politely and, after
  repeated attempts, offers a human transfer.
- **Emotional support** — rule-based detection of frustration, anger,
  anxiety, confusion, and refusal, each with an empathetic prefix layered
  onto the SOP-controlled reply (never replacing it).
- **LLM polish + guardrails** — Ollama chat client with a rewrite-only
  prompt and a leak-checker that rejects any hallucinated case ID, number,
  or forbidden term, falling back to the safe template.
- **Test UI** (`static/index.html`) — a call-console-styled single page:
  phase stepper, verified/case/intent/tone panel, an escalation banner, a
  debug log of internal SOP notes, and a dropdown to pick the representative
  consent scenario (`default`/`timeout`) for testing.
- **Automated tests** (`tests/test_scenarios.py`) — 14 tests covering the
  sample scenario, partial/refused verification, out-of-scope handling,
  both consent scenarios, an unregistered representative, grounded
  follow-up answers, and both email-summary choices. The suite runs
  identically whether or not Ollama is installed, since it always has the
  deterministic fallback path available.

## Project layout

```
insurance_sop_agent/
  app/
    config.py            env-driven settings
    models.py              session/turn dataclasses (framework-free)
    data_loader.py          fixture loading + lookups
    session_store.py        in-memory session store
    orchestrator.py         per-turn glue: gates -> phase dispatch -> LLM polish
    main.py                 FastAPI app (thin HTTP layer)
    nlu/
      extraction.py          regex/keyword PII + hint + signal extraction
      emotion.py              rule-based emotion + empathy templates
      scope.py                 in/out-of-scope guard
    sop/
      verify_id.py             VERIFY_ID phase + representative/consent flow
      resolve_intent.py         RESOLVE_INTENT phase + memory-based auto-resolve
      process_case.py           PROCESS_CASE phase (grounded Q&A)
      post_process.py           POST_PROCESS phase (email summary)
    llm/
      client.py                Ollama HTTP client
      compose.py                phrasing prompt + guardrail leak-checker
  static/index.html         browser test UI (single file, no build step)
  data/                     fixture data (policyholders, claims, etc.)
  tests/test_scenarios.py     unittest suite (also pytest-collectible)
  requirements.txt, .env.example, Dockerfile, docker-compose.yml
  scripts/run_dev.sh          convenience local-run script
```

## Setup & running

Requires **Python 3.10+** and, for natural phrasing, a running **Ollama**
server with the target model pulled. No paid API keys are used anywhere.

### Option A — plain Python (recommended when Ollama is already local)

```bash
cd insurance_sop_agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # edit OLLAMA_MODEL if the local tag differs
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or run `./scripts/run_dev.sh`, which does the above automatically.

Make sure Ollama is running with the model available:

```bash
ollama serve            # if not already running as a service
ollama list              # confirm the exact local tag, e.g. qwen3.5:4b
```

Then open **http://localhost:8000** for the test UI.

### Option B — Docker

The app container talks to Ollama running on the host machine (no model is
bundled into the image).

```bash
docker compose up --build
```

`docker-compose.yml` points `OLLAMA_HOST` at `http://host.docker.internal:11434`
and adds the `host.docker.internal` mapping for Linux hosts. Override any
setting via an `.env` file or `OLLAMA_HOST=... OLLAMA_MODEL=... docker
compose up --build`.

### Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Local Ollama server address |
| `OLLAMA_MODEL` | `qwen3.5:4b` | Model tag to call — run `ollama list` to confirm the exact local tag |
| `LLM_POLISH_ENABLED` | `true` | Set `false` to force the deterministic template engine only |
| `OLLAMA_TIMEOUT_SECONDS` | `20` | Per-call timeout before falling back to the template |
| `SOP_PORT` | `8000` | Server port |

### Running the tests

```bash
python3 -m unittest discover -s tests -v
# or, if pytest is installed:
pytest tests/ -v
```

The suite does not require Ollama to be running — it exercises the
deterministic SOP core directly.

## Using the test UI

- Type as the caller would speak. The left panel shows the live SOP state:
  phase stepper, verified flag, matched account, resolved case/intent,
  detected tone, and whether the last reply was phrased by the LLM or the
  template fallback.
- **Representative consent scenario** dropdown: pick `default` (approves
  after a couple of exchanges) or `timeout` (never approves, escalates)
  *before* saying you're calling on someone else's behalf, to exercise
  either branch of that flow.
- **Show SOP debug log**: internal one-line notes from each phase handler
  (which fields matched, why a topic was chosen, etc.).
- **End & start new call**: resets the session.

## Test scenarios

Each of these can be run as a fresh call (click **End & start new call**
first). They line up with `tests/test_scenarios.py`, so anything below can
also be found as an automated test of the same name.

### 1. One-shot verification + memory + resolution

> I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm
> calling about my denied healthcare claim from January. DOB is 1985-03-15,
> SSN last four is 4472.

Verifies Margaret Chen (3/3 matched fields) and, in the same reply, resolves
straight to claim `CL-2048` and explains the denial reason — using the
memory captured from that one utterance, with no re-asking. Case file
should show `Verified: Yes`, `Case: CL-2048`, `Intent: denial_question`.

### 2. Partial verification across multiple turns

Turn 1: `My name is Margaret Chen.`
Turn 2: `DOB is 1985-03-15`
Turn 3: `SSN last four is 4472`

Verification should stay pending after turns 1–2 (asking for one more
field each time) and complete on turn 3.

### 3. Frustration / refusal during verification

Turn 1: `My name is Margaret Chen.`
Turn 2: `I already told you who I am. This is ridiculous. Just tell me why
my claim was denied.`

`Caller tone` should read `refusing`. The reply should acknowledge the
frustration and explain *why* verification is required, but must still
withhold all claim details and keep asking for one more field — it should
never contain `CL-` or any claim content.

### 4. Representative flow — consent approved

Set the **Representative consent scenario** dropdown to `default` *before*
sending the first message.

Turn 1: `Hi, I'm calling on behalf of my mother Margaret Chen about her
claim.`
Turn 2: `Sure, I'll wait.`
Turn 3: `Any update?`

The agent should recognize the registered representative, start a simulated
consent request, poll it across turns, and verify (`Representative: Yes`,
`Account: P9`) once the scenario resolves to `approved` — without ever
disclosing claim details while `Verified` is still `No`.

### 5. Representative flow — consent times out

Set the dropdown to `timeout` first.

Turn 1: `I'm David Chen, calling on behalf of my mother.`
Turns 2+: `Any update?` (repeat a few times)

Consent should stay `pending` for a couple of polls and then resolve to
"couldn't confirm in time," offering a human transfer or asking the
policyholder to call in directly. `Verified` should remain `No` throughout.

### 6. Unregistered representative

> I'm calling on behalf of my friend John Smith about his policy.

Since there's no matching entry in `representatives.json`, the agent should
decline without confirming or denying whether an account exists for
"John Smith," and offer either a callback from the policyholder or a human
transfer.

### 7. Ambiguous case resolution

> My name is Margaret Chen, DOB 1985-03-15, SSN last four 4472, about a
> healthcare claim.

This verifies but only narrows to case *type* (healthcare), which still
matches two claims (`CL-2048` and `CL-2011`). The agent should list both and
ask which one — reply with `CL-2048 please` to resolve it.

### 8. Grounded document follow-up

After scenario 1 or 7 resolves to `CL-2048`, ask:

> How do I submit the pathology report?

The reply should reference the actual submission method from
`required_document_guideline.json` (the client portal) and the correct
case ID — nothing invented.

### 9. Missing-document alternative question

Following on from scenario 8:

> What if I can't get the original pathology report?

Should surface the documented alternative-evidence guidance for that
specific document, not a generic non-answer.

### 10. Mid-call topic switch to a different claim

Once verified (e.g. after scenario 1), send an explicit different case ID:

> Actually, can we talk about CL-2102 instead?

The agent should switch to that claim (if it belongs to the same verified
account) and give a fresh grounded presentation of it.

### 11. Out-of-scope questions + escalation offer

Turn 1: `What is reinforcement learning?`
Turn 2: `Tell me a joke instead.`

Turn 1 should get a polite decline and redirect back to the call's purpose.
Turn 2 (a second unrelated question) should additionally offer a human
transfer, without ever answering the off-topic question.

### 12. Explicit human transfer request

At any point in a call:

> Can I just talk to a real person?

Should immediately acknowledge and hand off, regardless of phase or
verification status.

### 13. Post-process — email summary accepted

After a case is resolved (e.g. scenario 1), end the case:

> That's all, thanks.

The agent should offer an email summary. Reply `Yes please` — it should
confirm the address it "sent" to and the call should move to `Wrap up` /
end.

### 14. Post-process — email summary declined

Same as above, but reply `No thanks` to the offer. The call should end
cleanly without sending anything, and `Reply phrased by` /
`email_decision` (visible via `/api/chat`'s JSON if inspecting network
calls) should read `skipped`.

### 15. Unknown / unmatched caller

> My name is Nobody Fake, DOB 2000-01-01, SSN last four 0000.

Since none of this matches any policyholder record, verification should
fail generically ("I wasn't able to match those details together...")
without ever hinting whether "Nobody Fake" does or doesn't have an account
— this is a deliberate privacy behavior, not a bug.

## Known limitations / possible next steps

- Session storage is in-memory (single process) — fine for a demo, swap for
  Redis/a database for multi-worker deployments.
- Emotion/scope/intent classification are rule-based by default for
  reliability with a small local model; they can be swapped for an
  LLM-classification call in `app/nlu/*.py` if more nuance is needed (the
  phase-level guardrails still apply either way).
- The email summary is simulated (returned in the API response, not
  actually delivered) — wire an SMTP/email API into
  `app/sop/post_process.py` for real delivery.
- Automated tests were run against the deterministic fallback path since
  the build/CI environment has no Ollama access; run the suite and a few
  manual calls locally to confirm phrasing quality against the actual
  model.
