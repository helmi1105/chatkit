"""Curriculum screening with bounded, rule-based follow-ups and auditable evidence."""
import json
import os
import uuid

from pydantic import BaseModel, Field

from app.providers import run_structured
from app.decision_policy import build_learner_state, decide_next_action, evidence_status, select_followups, evidence_task
from app.schemas import QcmQuestion, normalize_questions


# Keep enough doctrine for a complete chapter while avoiding the latency and
# distraction caused by sending the largest possible source bundle every time.
# Deployments can raise this value for unusually dense chapters without a code
# change. Focused one-KC repairs retain their smaller 6,000-character context.
DIAGNOSTIC_CONTEXT_MAX_CHARS = max(
    6000, int(os.getenv('DIAGNOSTIC_CONTEXT_MAX_CHARS', '20000'))
)
DIAGNOSTIC_REPAIR_MAX_ATTEMPTS = max(
    1, int(os.getenv('DIAGNOSTIC_REPAIR_MAX_ATTEMPTS', '2'))
)


class DiagnosticRound(BaseModel):
    scenario: str = ''
    questions: list[QcmQuestion] = Field(default_factory=list)


class DiagnosticSingleQuestion(BaseModel):
    """Required output for a focused one-KC repair.

    Unlike DiagnosticRound, this cannot silently validate a response that
    omitted the question field and turn it into an empty list.
    """
    scenario: str = ''
    question: QcmQuestion


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
    @staticmethod
    def _server_owned_question(raw_question, kc_id, source_pages=()):
        """Attach scoring metadata chosen by the policy, never by the model.

        This is safe only for a generation request containing exactly one KC;
        callers enforce that invariant before using the returned question.
        """
        if isinstance(raw_question, BaseModel):
            raw_question = raw_question.model_dump()
        if not isinstance(raw_question, dict):
            return None
        question = dict(raw_question)
        # In a one-page, one-KC request the source page is unambiguous.  Small
        # models sometimes omit it even though the generated content is valid.
        if len(source_pages) == 1:
            try:
                returned_page = int(question.get('page'))
            except (TypeError, ValueError):
                returned_page = None
            if returned_page != source_pages[0]:
                question['page'] = source_pages[0]
        question.update(
            kc_id=kc_id,
            role='current',
            assessed_kc_ids=[kc_id],
        )
        normalized = normalize_questions([question], kc_id)
        return normalized[0] if len(normalized) == 1 else None

    async def _regenerate_diagnostic_questions_individually(
        self, sess, group_title, stage, tasks, ctx, *, max_attempts=3
    ):
        """Recover a malformed grouped response without weakening scoring.

        Each call contains one and only one curriculum notion.  The server can
        therefore stamp the requested KC metadata deterministically while the
        model remains responsible only for the pedagogical content.
        """
        questions = []
        scenario = ''
        previous = {
            str(item.get('question') or '').strip().casefold()
            for values in sess.diagnostic_evidence.values()
            for item in values
        }
        for task in tasks:
            kid = task['kc_id']
            pages = self.graph.kc_pages(self.graph.nodes[kid])
            source = self.doc.pages_text(pages, max_chars=6000)
            generated = None
            rejection_reasons = []
            forbidden_questions = sorted(
                text for text in previous if text
            )[-8:]
            for attempt in range(max_attempts):
                self._charge_budget(sess)
                out = await run_structured(
                    'Hierarchical-diagnostic-single-kc',
                    'Tu es formateur. Utilise uniquement les extraits fournis. '
                    'Produis exactement un QCM autonome, avec quatre choix distincts et une seule reponse correcte.',
                    f"Chapitre : {group_title}. Etape : {stage}.\n"
                    f"Notion unique : {json.dumps(task, ensure_ascii=False)}\n"
                    f"Tentative : {attempt + 1}. Questions deja posees, interdites de repetition : "
                    f"{json.dumps(forbidden_questions, ensure_ascii=False)}\n"
                    "Produis exactement UNE nouvelle question dans le champ JSON obligatoire `question`. "
                    "Change nettement l'exemple et l'enonce. "
                    "answer=lettre A/B/C/D, explanation=justification avec page, page=source. "
                    "Le serveur ajoutera les identifiants techniques; concentre-toi sur le contenu.\n"
                    + source,
                    DiagnosticSingleQuestion,
                    ctx,
                )
                raw_questions = []
                single_question = getattr(out, 'question', None)
                if single_question is not None:
                    raw_questions.append(single_question)
                # Compatibility for mocked/older provider adapters while
                # deployments move to the required singular schema.
                raw_questions.extend(list(getattr(out, 'questions', []) or []))
                if not raw_questions:
                    rejection_reasons.append('aucune question')
                    continue
                for raw_question in raw_questions:
                    candidate = self._server_owned_question(raw_question, kid, pages)
                    if not candidate:
                        rejection_reasons.append('format QCM invalide')
                        continue
                    if candidate['page'] not in pages:
                        rejection_reasons.append('page source incorrecte')
                        continue
                    if candidate['text'].strip().casefold() in previous:
                        rejection_reasons.append('question repetee')
                        continue
                    generated = candidate
                    scenario = scenario or str(out.scenario or '').strip()
                    break
                if generated is not None:
                    break
            if generated is None:
                detail = ', '.join(dict.fromkeys(rejection_reasons)) or 'sortie vide'
                raise RuntimeError(
                    f'Diagnostic inexploitable : question absente ou invalide pour {kid} ({detail}).'
                )
            questions.append(generated)
        for number, question in enumerate(questions, start=1):
            question['number'] = number
        return questions, scenario

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
        source = self.doc.pages_text(
            pages, max_chars=DIAGNOSTIC_CONTEXT_MAX_CHARS
        )
        print(
            f"[diagnostic] generation chapter={group['module_id']} "
            f"kcs={len(ids)} pages={len(pages)} context_chars={len(source)} "
            f"context_limit={DIAGNOSTIC_CONTEXT_MAX_CHARS}"
        )
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
            + "\nChaque question declare role=current et assessed_kc_ids=[son kc_id unique].\n"
            + source, DiagnosticRound, ctx)
        questions = normalize_questions(out.questions, '')
        if any(q.get('role') != 'current' or q.get('assessed_kc_ids') != [q['kc_id']]
               for q in questions):
            self._charge_budget(sess)
            repair = await run_structured('Hierarchical-diagnostic-attribution-repair',
                'Tu es formateur. Utilise uniquement les extraits fournis. Une question par notion, quatre choix distincts.',
                f"Chapitre : {group['title']}. Etape : {stage}. Taches et preuves precedentes : "
                f"{json.dumps(tasks, ensure_ascii=False)}. Regenerer exactement une question par KC : {ids}. "
                "Respecter evidence_type, changer les exemples et ne pas repeter les questions precedentes. Chaque question doit declarer "
                "role=current et assessed_kc_ids contenant uniquement son kc_id. "
                "Inclure un numero de page propre a chaque KC.\n" + source,
                DiagnosticRound, ctx)
            out = repair
            questions = normalize_questions(out.questions, '')
        mapping_ok = len(questions) == len(ids) and {q['kc_id'] for q in questions} == set(ids)
        attribution_ok = mapping_ok and all(
            q.get('role') == 'current' and q.get('assessed_kc_ids') == [q['kc_id']]
            for q in questions
        )
        # When a KC has one authoritative page, a missing/wrong page number is
        # metadata we can repair without spending another model call. The model
        # received only that page's text, so there is no page choice to infer.
        if mapping_ok and attribution_ok:
            for question in questions:
                source_pages = self.graph.kc_pages(self.graph.nodes[question['kc_id']])
                if len(source_pages) == 1 and question.get('page') != source_pages[0]:
                    question['page'] = source_pages[0]
        previous = {
            str(item.get('question') or '').strip().casefold()
            for values in sess.diagnostic_evidence.values()
            for item in values
        }
        invalid_content_ids = {
            q['kc_id'] for q in questions
            if q['text'].strip().casefold() in previous
            or q.get('page') not in self.graph.kc_pages(self.graph.nodes[q['kc_id']])
        } if mapping_ok and attribution_ok else set()
        if mapping_ok and attribution_ok and invalid_content_ids:
            # Keep valid grouped questions and regenerate only invalid KCs.
            # Usually this is one extra call, even for a multi-question chapter.
            repair_tasks = [t for t in tasks if t['kc_id'] in invalid_content_ids]
            repaired, _ = await self._regenerate_diagnostic_questions_individually(
                sess, group['title'], stage, repair_tasks, ctx,
                max_attempts=1,
            )
            print(
                '[diagnostic] focused_repair '
                f"kcs={','.join(sorted(invalid_content_ids))} attempts=1"
            )
            by_kc = {q['kc_id']: q for q in questions
                     if q['kc_id'] not in invalid_content_ids}
            by_kc.update({q['kc_id']: q for q in repaired})
            questions = [by_kc[kid] for kid in ids]
            for number, question in enumerate(questions, start=1):
                question['number'] = number
        elif not mapping_ok or not attribution_ok:
            # Structural failures need regeneration for every requested KC.
            print(
                '[diagnostic] structural_repair '
                f"kcs={len(tasks)} max_attempts={DIAGNOSTIC_REPAIR_MAX_ATTEMPTS}"
            )
            questions, fallback_scenario = await self._regenerate_diagnostic_questions_individually(
                sess, group['title'], stage, tasks, ctx,
                max_attempts=DIAGNOSTIC_REPAIR_MAX_ATTEMPTS,
            )
            if fallback_scenario:
                out.scenario = fallback_scenario
        if len(questions) != len(ids) or {q['kc_id'] for q in questions} != set(ids):
            raise RuntimeError('Diagnostic inexploitable : correspondance question/notion incorrecte.')
        if any(q.get('role') != 'current' or q.get('assessed_kc_ids') != [q['kc_id']]
               for q in questions):
            raise RuntimeError('Diagnostic inexploitable : role ou notions evaluees manquants/incoherents.')
        previous = {o['question'].strip().casefold() for v in sess.diagnostic_evidence.values() for o in v}
        if any(q['text'].strip().casefold() in previous for q in questions):
            raise RuntimeError('Diagnostic inexploitable : question répétée après régénération.')
        if any(q['page'] not in self.graph.kc_pages(self.graph.nodes[q['kc_id']])
               for q in questions):
            raise RuntimeError('Diagnostic inexploitable : page source absente après régénération.')
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
