"""Curriculum screening with bounded, rule-based follow-ups and auditable evidence."""
import json
import uuid

from pydantic import BaseModel, Field

from app.providers import run_structured
from app.decision_policy import build_learner_state, decide_next_action, evidence_status, select_followups, evidence_task
from app.schemas import QcmQuestion, normalize_questions


class DiagnosticRound(BaseModel):
    scenario: str = ''
    questions: list[QcmQuestion] = Field(default_factory=list)


def curriculum_groups(graph):
    """Contiguous course groups, including root KCs; sequence is NOT prerequisite."""
    groups = []
    for kid in graph.kc_ids():
        mid = graph.module_of(kid) or 'ROOT'
        if not groups or groups[-1]['module_id'] != mid:
            groups.append({'module_id': mid, 'title': graph.module_title(kid), 'kc_ids': []})
        groups[-1]['kc_ids'].append(kid)
    return groups


class HierarchicalDiagnostic:
    async def _start_hierarchical(self, sess, ctx):
        if sess.diagnostic_stage == 'learning' and not sess.diagnostic_done:
            return [self._text('Le diagnostic est en pause : travaillez la notion en cours, puis il reprendra.'),
                    self._actions(self._next_actions(sess))]
        if sess.diagnostic_mode == 'hierarchical' and not sess.diagnostic_done and sess.diagnostic_stage:
            if sess.phase == 'waiting_answers':
                from app.orchestrator import qcm_widget_data
                return [self._text('Terminez cette étape du diagnostic.\n\n' + sess.diagnostic_scenario),
                        {'type': 'qcm', 'data': qcm_widget_data('Diagnostic en cours', list(sess.quiz_questions.values()))}]
            return await self._continue_hierarchical(sess, ctx)
        # A new diagnostic must not erase validated practice history.
        from app.orchestrator import DIAGNOSTIC_MAX_QUESTIONS
        ids = self.graph.kc_ids()
        if not ids or DIAGNOSTIC_MAX_QUESTIONS < 1:
            raise ValueError('Le diagnostic nécessite un cours et un budget positif.')
        sess.diagnostic_mode = 'hierarchical'
        sess.diagnostic_stage = 'screening'
        sess.diagnostic_kc_ids = ids
        sess.diagnostic_groups = curriculum_groups(self.graph)
        sess.diagnostic_group_index = 0
        sess.diagnostic_policy_version = 2
        sess.diagnostic_status_by_kc = {}
        sess.diagnostic_scenario = ''
        sess.diagnostic_question_limit = DIAGNOSTIC_MAX_QUESTIONS
        sess.diagnostic_evidence = {}
        sess.diagnostic_taught_kc_ids = []
        sess.diagnostic_round = 0
        sess.diagnostic_done = False
        sess.diagnostic_profile = {}
        sess.diagnostic_raw_score_by_kc = {}
        sess.scope = 'diagnostic'
        self._clear_quiz(sess)
        return await self._continue_hierarchical(sess, ctx)

    async def _continue_hierarchical(self, sess, ctx):
        # Upgrade persisted unfinished pilot sessions without discarding their answers.
        if not sess.diagnostic_groups:
            sess.diagnostic_groups = curriculum_groups(self.graph)
            sess.diagnostic_kc_ids = self.graph.kc_ids()
            sess.diagnostic_policy_version = 2
        sess.scope = 'diagnostic'
        sess.diagnostic_status_by_kc = {k: evidence_status(sess.diagnostic_evidence.get(k, [])) for k in self.graph.kc_ids()}
        state = build_learner_state(sess, self.graph)
        decision = decide_next_action(state, sess.diagnostic_groups)
        self._record_event(sess, ctx, {'event': 'pedagogical_decision',
            'phase': 'diagnostic', 'learner_state': state,
            'curriculum': sess.diagnostic_groups, 'decision': decision})
        if decision['action'] in {'teach', 'validate'}:
            return await self._pause_for_learning(sess, decision['target_kc'], ctx,
                                                  challenge=decision['action'] == 'validate')
        for index in decision['completed_group_indices']:
            group = sess.diagnostic_groups[index]
            statuses = {k: sess.diagnostic_status_by_kc[k] for k in group['kc_ids']}
            self._record_event(sess, ctx, {'event': 'diagnostic_chapter_completed',
                'module_id': group['module_id'], 'statuses': statuses,
                'decision': 'skip_introductory_lessons' if all(s == 'supported' for s in statuses.values()) else 'queue_targeted_learning',
                'next_action': 'screen_next_curriculum_group'})
            sess.diagnostic_group_index = index + 1
        if decision['action'] == 'finish':
            return await self._finish_hierarchical(sess, ctx)
        sess.diagnostic_group_index = decision['group_index']
        group = sess.diagnostic_groups[sess.diagnostic_group_index]
        ids = [task['kc_id'] for task in decision['tasks']]
        stage = 'screening' if all(not sess.diagnostic_evidence.get(k) for k in ids) else 'followup'
        tasks = []
        for kid in ids:
            observations = sess.diagnostic_evidence.get(kid, [])
            evidence_type, reason = evidence_task(observations)
            tasks.append({'kc_id': kid, 'title': self.graph.nodes[kid].title,
                          'evidence_type': evidence_type, 'selection_reason': reason,
                          'previous_evidence': observations})
        pages = sorted({p for kid in ids for p in self.graph.kc_pages(self.graph.nodes[kid])})
        source = self.doc.pages_text(pages, max_chars=30000)
        self._charge_budget(sess)
        out = await run_structured('Hierarchical-diagnostic',
            'Tu es formateur. Utilise uniquement les extraits fournis. Une réponse correcte par QCM, quatre choix distincts. '
            'Chaque question évalue une seule notion identifiée; ne déduis pas la maîtrise des autres notions.',
            f"Chapitre : {group['title']}. Étape : {stage}.\n"
            f"Tâches : {json.dumps(tasks, ensure_ascii=False)}\n"
            "Produis exactement une question par kc_id demandé. Pour le screening, crée une courte situation opérationnelle commune, "
            "sans révéler les réponses, et des questions séparément notées sur ses différents aspects. "
            "Chaque question doit être compréhensible avec cette situation. Pour les confirmations, change l'exemple et teste l'application. "
            "Pour une discrimination ciblée, distingue la bonne règle du distracteur précédemment choisi. "
            "Ne répète aucune question précédente. answer=lettre, explanation=justification avec page, page=source.\n"
            "Respecte evidence_type : recognition=identification, application=utilisation dans un nouveau cas, "
            "discrimination=choix entre la règle et la confusion observée, transfer=nouveau contexte. "
            "N'affirme pas qu'une erreur prouve une misconception. Chaque énoncé doit rester autonome.\n"
            + source, DiagnosticRound, ctx)
        questions = normalize_questions(out.questions, '')
        if len(questions) != len(ids) or {q['kc_id'] for q in questions} != set(ids):
            raise RuntimeError('Diagnostic inexploitable : correspondance question/notion incorrecte.')
        previous = {o['question'].strip().casefold() for v in sess.diagnostic_evidence.values() for o in v}
        if any(q['text'].strip().casefold() in previous or
               q['page'] not in self.graph.kc_pages(self.graph.nodes[q['kc_id']]) for q in questions):
            raise RuntimeError('Diagnostic inexploitable : question répétée ou source absente.')
        task_by_kc = {t['kc_id']: t for t in tasks}
        for q in questions:
            task = task_by_kc[q['kc_id']]
            q.update(evidence_type=task['evidence_type'], selection_reason=task['selection_reason'])
        sess.diagnostic_scenario = out.scenario
        sess.diagnostic_round += 1
        sess.diagnostic_stage = stage
        sess.quiz_id = uuid.uuid4().hex[:12]
        sess.quiz_questions = {str(q['number']): q for q in questions}
        sess.phase = 'waiting_answers'
        sess.last_tutor_action = 'hierarchical_diagnostic_round'
        event = 'diagnostic_started' if sess.diagnostic_round == 1 else 'diagnostic_round_started'
        self._record_event(sess, ctx, {'event': event, 'mode': 'hierarchical', 'module_id': group['module_id'],
            'round': sess.diagnostic_round, 'stage': stage, 'kc_ids': ids, 'tasks': tasks,
            'n_questions': len(questions), 'quiz_id': sess.quiz_id, 'source_pages': pages,
            'question_limit': sess.diagnostic_question_limit, 'scenario': out.scenario})
        from app.orchestrator import qcm_widget_data
        return [self._text(f"Diagnostic — {group['title']}, étape {sess.diagnostic_round}. "
                          "Les notions déjà étayées ne reçoivent plus de questions. Le corrigé sera présenté à la fin.\n\n" + out.scenario),
                {'type': 'qcm', 'data': qcm_widget_data('Diagnostic du chapitre', questions)}]

    async def _submit_hierarchical(self, sess, answers, ctx):
        if any(str(answers.get(int(n), '')).upper() not in ('A', 'B', 'C', 'D') for n in sess.quiz_questions):
            return [self._text('Répondez à toutes les questions avant de valider cette étape.')]
        score, _, items = self._score(sess, answers)
        for item in items:
            q = sess.quiz_questions[str(item['number'])]
            item.update(evidence_type=q.get('evidence_type', 'legacy_unspecified'),
                        after_instruction=bool(sess.diagnostic_taught_kc_ids),
                        selection_reason=q.get('selection_reason', 'legacy_pilot'),
                        scenario=sess.diagnostic_scenario)
            sess.diagnostic_evidence.setdefault(item['kc_id'], []).append(
                {**item, 'round': sess.diagnostic_round, 'quiz_id': sess.quiz_id})
        self._record_event(sess, ctx, {'event': 'diagnostic_round_submitted', 'round': sess.diagnostic_round,
            'quiz_id': sess.quiz_id, 'score': score, 'items': items, 'evidence_role': 'screening_only'})
        self._clear_quiz(sess)
        # Saved even if generation of the following round fails; start resumes here.
        await self._save_session(sess)
        return await self._continue_hierarchical(sess, ctx)

    async def _pause_for_learning(self, sess, kid, ctx, *, challenge=False):
        sess.diagnostic_stage = 'learning'
        sess.diagnostic_validation_challenge = challenge
        sess.scope = 'practice'
        sess.current_kc_id = kid
        sess.can_advance = False
        sess.validated_kc_id = None
        sess.pending_next_kc_id = None
        sess.pending_module_id = None
        sess.module_gate_locked = False
        sess.pending_module_retry = False
        sess.pending_hint_ladder = []
        sess.hint_index = 0
        sess.current_micro_lesson = ''
        self._clear_quiz(sess)
        mistakes = [o for o in sess.diagnostic_evidence[kid] if not o['correct']]
        sess.last_mistakes_summary = '\n'.join(
            f"Q{o['number']}: {o['question']} | répondu {o['learner_choice']} au lieu de {o['correct_choice']}" for o in mistakes)
        self._record_event(sess, ctx, {'event': 'diagnostic_paused_for_learning', 'kc_id': kid,
            'status': sess.diagnostic_status_by_kc[kid], 'group_index': sess.diagnostic_group_index,
            'evidence': sess.diagnostic_evidence[kid], 'next_action': 'validation_quiz' if challenge else 'lesson_then_practice'})
        await self._save_session(sess)
        if challenge:
            quiz = await self._start_practice(sess, ctx)
            return [self._text('Diagnostic réussi pour cette notion. Passez le quiz de validation sans leçon préalable.'), *quiz]
        corrections = '\n'.join(f"Q{o['number']} : {o['correct_letter']}) {o['correct_choice']}. {o['explanation']}" for o in mistakes)
        lesson = await self._show_lesson(sess, ctx)
        return [self._text('Diagnostic en pause pour travailler « ' + self.graph.nodes[kid].title +
                          ' ». Après réussite du quiz, vous pourrez reprendre le diagnostic.\n\n' + corrections), *lesson]

    async def _finish_hierarchical(self, sess, ctx):
        statuses = {k: evidence_status(sess.diagnostic_evidence.get(k, [])) for k in self.graph.kc_ids()}
        assessed = [k for k in sess.diagnostic_kc_ids if sess.diagnostic_evidence.get(k)]
        weak = [k for k in assessed if any(not o['correct'] for o in sess.diagnostic_evidence[k])]
        sess.weak_queue = weak
        sess.diagnostic_raw_score_by_kc = {k: sum(o['correct'] for o in sess.diagnostic_evidence[k]) /
                                          len(sess.diagnostic_evidence[k]) for k in assessed}
        # No artificial probabilities and no changes to mastery or validated IDs.
        sess.diagnostic_status_by_kc = statuses
        sess.diagnostic_learning_queue = [k for k in self.graph.kc_ids()
                                         if statuses[k] != 'supported' and k not in sess.validated_kc_ids]
        sess.current_kc_id = next(iter(sess.diagnostic_learning_queue), None)
        sess.can_advance = False
        sess.validated_kc_id = None
        sess.pending_next_kc_id = None
        sess.pending_module_id = None
        sess.module_gate_locked = False
        sess.pending_module_retry = False
        sess.pending_hint_ladder = []
        sess.hint_index = 0
        sess.scope = 'practice'
        sess.diagnostic_done = True
        sess.diagnostic_stage = 'finished'
        sess.current_micro_lesson = ''
        sess.last_mistakes_summary = ''
        self._clear_quiz(sess)
        items = [o for k in assessed for o in sess.diagnostic_evidence[k]]
        score = sum(o['correct'] for o in items) / max(1, len(items))
        reason = 'question_budget' if len(items) >= sess.diagnostic_question_limit else 'evidence_rule'
        self._record_event(sess, ctx, {'event': 'diagnostic_submitted', 'mode': 'hierarchical',
            'evidence_role': 'screening_only', 'mastery_update': 'none_from_diagnostic',
            'mastery_before': dict(sess.mastery), 'mastery_after': dict(sess.mastery),
            'statuses': statuses, 'overall_score': score, 'weak_queue': weak,
            'diagnostic_raw_score_by_kc': sess.diagnostic_raw_score_by_kc,
            'n_questions': len(items), 'stop_reason': reason, 'policy_version': 2,
            'directly_assessed_kc_ids': assessed, 'inferred_kc_ids': [],
            'learning_queue': sess.diagnostic_learning_queue,
            'skipped_intro_kc_ids': [k for k in statuses if statuses[k] == 'supported']})
        labels = {'supported': 'réussites confirmées au diagnostic', 'uncertain': 'à vérifier',
                  'needs_practice': 'à travailler', 'unassessed': 'non évaluée'}
        summary = '\n'.join(f"• {self.graph.nodes[k].title} : {labels[statuses[k]]}" for k in sess.diagnostic_kc_ids)
        # Text corrections preserve original quiz/round numbering and avoid wrong report IDs.
        corrections = '\n'.join(f"Étape {o['round']}, Q{o['number']} : {o['question']}\n"
            f"Votre réponse : {o['learner_letter']} — {o['learner_choice']}. "
            f"Réponse correcte : {o['correct_letter']} — {o['correct_choice']}. {o['explanation']}" for o in items)
        next_text = (f"Leçon ciblée proposée : {self.graph.nodes[sess.current_kc_id].title}." if sess.current_kc_id else
                     'Aucune leçon introductive nécessaire selon ces observations. Vous pouvez consulter votre progression ou poser une question.')
        blocks = [self._text(f"Diagnostic terminé : {sum(o['correct'] for o in items)}/{len(items)}.\n"
                          + summary + '\nLes notions étayées sont dispensées de leçon introductive, sans validation automatique.\n' + next_text),
                self._text('Corrigé du diagnostic\n\n' + corrections),
                self._actions(([('Lire la leçon ciblée', 'revoir la leçon')] if sess.current_kc_id else []) +
                              [('Ma progression', 'ma progression'), ('Poser une question', 'question')])]

        blocks.extend(await self._completion_checkpoint(sess, ctx))
        return blocks

    def _learning_next(self, sess, current):
        if sess.diagnostic_mode == 'hierarchical' and sess.diagnostic_policy_version >= 2 and sess.diagnostic_done:
            return next((k for k in sess.diagnostic_learning_queue if k != current and k not in sess.validated_kc_ids), None)
        return self.graph.next_kc(current or '')
