"""Replay new pedagogical_decision JSONL events without model calls.

Optional expert annotation per event:
acceptable_decisions: [{"action": "teach", "target_kc": "g_k2"}, ...]
"""
import argparse
import json
from app.decision_policy import decide_next_action, POLICY_VERSION


def evaluate(events):
    total = matches = labelled = accepted = unsupported = 0
    for event in events:
        if event.get('event') != 'pedagogical_decision':
            continue
        if event['decision'].get('policy_version') != POLICY_VERSION:
            unsupported += 1
            continue
        actual = decide_next_action(event['learner_state'], event['curriculum'],
                                    practice_result=event.get('practice_result'))
        total += 1
        matches += actual == event['decision']
        gold = event.get('acceptable_decisions')
        if gold:
            labelled += 1
            accepted += any(actual['action'] == g['action'] and actual['target_kc'] == g['target_kc'] for g in gold)
    return dict(replayed=total, exact_replay_matches=matches, unsupported_versions=unsupported,
                expert_labelled=labelled, expert_agreement=accepted / labelled if labelled else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence_jsonl')
    args = parser.parse_args()
    with open(args.evidence_jsonl, encoding='utf-8') as source:
        print(json.dumps(evaluate(json.loads(line) for line in source if line.strip()), indent=2))
