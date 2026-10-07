"""Pure, replayable baseline policy. No model calls or session mutations.

Confidence is deliberately null: these rules are not calibrated probabilities.
Curriculum sequence is used for ordering, never as prerequisite evidence.
"""
from copy import deepcopy

POLICY_VERSION = "baseline-rules-v1"

def evidence_status(observations):
    if not observations:
        return 'unassessed'
    if len(observations) < 2:
        return 'uncertain'
    results = [bool(o['correct']) for o in observations]
    types = {o.get('evidence_type') for o in observations if o['correct']}
    if all(results) and 'recognition' in types and types.intersection({'application', 'transfer'}):
        return 'supported'
    if results.count(False) >= 2:
        return 'needs_practice'
    return 'uncertain'


def select_followups(kc_ids, evidence, remaining):
    # Every screened KC gets an independent confirmation; mixed evidence gets a probe.
    candidates = [k for k in kc_ids if len(evidence.get(k, [])) < 2 or
                  (len(evidence.get(k, [])) < 3 and evidence_status(evidence[k]) == 'uncertain')]
    # Prioritize contradictions, then first failures, then confirmations/unassessed.
    def priority(k):
        obs = evidence.get(k, [])
        return 0 if len(obs) >= 2 else (1 if obs and not obs[-1]['correct'] else 2)
    return sorted(candidates, key=priority)[:max(0, remaining)]


def evidence_task(observations):
    if not observations:
        return 'recognition', 'chapter_screening'
    if all(o['correct'] for o in observations):
        return 'application', 'confirm_recognition_in_new_scenario'
    if len(observations) == 1 or not observations[-1]['correct']:
        return 'discrimination', 'investigate_observed_wrong_choice'
    return 'transfer', 'resolve_contradictory_evidence'



def build_learner_state(sess, graph):
    """Snapshot observed evidence separately from inferred labels and practice scores."""
    kcs = {}
    for kid in graph.kc_ids():
        observations = deepcopy(sess.diagnostic_evidence.get(kid, []))
        kcs[kid] = {
            'diagnostic_evidence': observations,
            'diagnostic_status': evidence_status(observations),
            'validated': kid in sess.validated_kc_ids,
            'practice_score_estimate': sess.mastery.get(kid),
            'last_assessment_score': sess.last_score_by_kc.get(kid),
            'practice_attempts': sess.attempts_by_kc.get(kid, 0),
            'recorded_confusions': list(sess.misconceptions.get(kid, [])),
            'known_objectives': list(sess.kc_essential_targets.get(kid, [])),
            'practice_evidence': deepcopy(sess.practice_evidence_by_kc.get(kid, [])),
            'confidence': None,
        }
    return {
        'kcs': kcs,
        'diagnostic_kc_ids': list(sess.diagnostic_kc_ids),
        'completed_learning_kc_ids': list(sess.diagnostic_taught_kc_ids),
        'group_index': sess.diagnostic_group_index,
        'question_limit': sess.diagnostic_question_limit,
        'questions_answered': sum(len(v) for v in sess.diagnostic_evidence.values()),
        'current_kc_id': sess.current_kc_id,
        'validation_challenge': sess.diagnostic_validation_challenge,
    }


def decide_next_action(state, curriculum, *, practice_result=None):
    """Choose a diagnostic route or a post-practice action from an immutable snapshot.

    curriculum is the ordered list of chapter groups. No prerequisite links are
    inferred. practice_result supplies the actual score and configured thresholds.
    validate means administer a quiz; advance means its passing criterion was met.
    """
    def result(action, reason, target=None, **extra):
        return dict(policy_version=POLICY_VERSION, action=action, reason=reason,
                    target_kc=target, confidence=None, **extra)
    if practice_result is not None:
        score = practice_result['score']
        kid = state['current_kc_id']
        if score >= practice_result['threshold']:
            return result('advance', 'practice_threshold_reached', kid)
        if state['validation_challenge'] or score < practice_result['remediation_below']:
            return result('remediate', 'failed_challenge' if state['validation_challenge'] else 'low_practice_score', kid)
        return result('review', 'practice_below_threshold', kid)
    ids = [k for k in state['diagnostic_kc_ids'] if k in state['kcs']]
    for kid in ids:
        kc = state['kcs'][kid]
        if kc['validated'] or kid in state['completed_learning_kc_ids']:
            continue
        status = kc['diagnostic_status']
        if status == 'needs_practice' or (status == 'uncertain' and len(kc['diagnostic_evidence']) >= 3):
            return result('teach', 'repeated_errors' if status == 'needs_practice' else 'unresolved_after_three_probes', kid)
    for kid in ids:
        kc = state['kcs'][kid]
        if kc['diagnostic_status'] == 'supported' and not kc['validated']:
            return result('validate', 'recognition_and_application_supported', kid)
    remaining = state['question_limit'] - state['questions_answered']
    if remaining <= 0:
        return result('finish', 'question_budget', completed_group_indices=[])
    evidence = {k: v['diagnostic_evidence'] for k, v in state['kcs'].items()}
    completed = []
    for index in range(state['group_index'], len(curriculum)):
        candidates = [k for k in curriculum[index]['kc_ids']
                      if k in evidence and k not in state['completed_learning_kc_ids']]
        selected = select_followups(candidates, evidence, remaining)
        if selected:
            tasks = [dict(kc_id=k, evidence_type=evidence_task(evidence[k])[0],
                          selection_reason=evidence_task(evidence[k])[1]) for k in selected]
            return result('probe', 'bounded_curriculum_followup', selected[0],
                          tasks=tasks, group_index=index, completed_group_indices=completed)
        completed.append(index)
    return result('finish', 'evidence_rule', completed_group_indices=completed)
