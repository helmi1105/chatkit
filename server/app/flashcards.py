"""Optional text revision; reveal/review events are not mastery evidence."""
import uuid
from typing import Literal
from pydantic import BaseModel, Field
from app.providers import run_structured
from app.widgets import its_widgets as W
from app.svg_answer import SVG_INSTRUCTIONS, validate_svg, inline_illustration_widget
from chatkit.widgets import Card, Text, Title, Caption, Row, Col, Button, Divider
from chatkit.actions import ActionConfig


class Flashcard(BaseModel):
    question: str
    answer: str
    page: int
    svg: str = ''
    svg_side: Literal['front', 'back'] = 'back'


class FlashcardPack(BaseModel):
    cards: list[Flashcard] = Field(default_factory=list)


class Flashcards:
    async def _flashcards(self, sess, command, ctx):
        kc = self._kc(sess.current_kc_id)
        if sess.phase == 'waiting_answers':
            return [self._text('Terminez le quiz avant de consulter les cartes de révision.')]
        if not kc or (not sess.diagnostic_done and sess.diagnostic_stage != 'learning'):
            return [self._text('Les cartes sont disponibles pendant l’apprentissage d’une notion, après son diagnostic.')]
        deck = sess.flashcard_deck
        if deck:
            deck.setdefault('queue', list(range(len(deck['cards']))))
            deck.setdefault('ratings', [])
            deck.setdefault('latest_ratings', {})
        parts = command.split()
        if parts[0] == 'fiche':
            if (len(parts) != 4 or not deck or deck.get('kc_id') != kc.id
                    or parts[2] != deck['id'] or parts[3] != str(deck['index'])):
                return [self._text('Cette carte n’est plus active. Ouvrez les cartes de révision de la notion en cours.')]
            action = parts[1]
            if deck['index'] >= len(deck['queue']):
                return [self._text('Cette révision est terminée. Ouvrez un nouveau paquet de cartes.')]
            card_id = deck['queue'][deck['index']]
            if action not in {'reveler', 'revoir', 'suivante', 'encore', 'difficile', 'sais'}:
                return [self._text('Action de révision inconnue.')]
            if action in {'revoir', 'suivante', 'encore', 'difficile', 'sais'} and not deck['revealed']:
                return [self._text('Révélez d’abord la réponse.')]
            if action == 'reveler' and not deck['revealed']:
                deck['revealed'] = True
                self._record_event(sess, ctx, {'event': 'flashcard_revealed', 'kc_id': kc.id,
                    'deck_id': deck['id'], 'card_index': card_id, 'turn': deck['index'], 'evidence_role': 'content_viewed_only'})
            elif action == 'revoir' and deck['index'] not in deck['review_indices']:
                deck['review_indices'].append(deck['index'])
                self._record_event(sess, ctx, {'event': 'flashcard_marked_for_review', 'kc_id': kc.id,
                    'deck_id': deck['id'], 'card_index': deck['index'], 'evidence_role': 'self_report_only'})
            elif action in {'encore', 'difficile', 'sais'}:
                rating = {'encore': 'again', 'difficile': 'hard', 'sais': 'know'}[action]
                record = {'card_index': card_id, 'turn': deck['index'], 'rating': rating}
                deck['ratings'].append(record)
                deck['latest_ratings'][str(card_id)] = rating
                if rating != 'know':
                    if card_id not in deck['review_indices']:
                        deck['review_indices'].append(card_id)
                    # Again returns after one intervening card; Hard at the end.
                    position = min(deck['index'] + 2, len(deck['queue'])) if rating == 'again' else len(deck['queue'])
                    deck['queue'].insert(position, card_id)
                else:
                    deck['review_indices'] = [i for i in deck['review_indices'] if i != card_id]
                self._record_event(sess, ctx, {'event': 'flashcard_rated', 'kc_id': kc.id,
                    'deck_id': deck['id'], **record, 'evidence_role': 'self_report_only'})
                deck['index'] += 1
                deck['revealed'] = False
            elif action == 'suivante':
                deck['index'] += 1
                deck['revealed'] = False
        elif not deck or deck.get('kc_id') != kc.id or deck['index'] >= len(deck['queue']):
            pages = self.graph.kc_pages(kc)
            source = self.doc.pages_text(pages, max_chars=9000)
            if not source.strip() or not pages:
                return [self._text('Source indisponible pour préparer ces cartes.')]
            self._charge_budget(sess)
            out = await run_structured('Flashcards',
                'Crée des cartes de révision en français, uniquement à partir des extraits. '
                'Une question autonome courte (maximum 180 caractères) et une réponse concise (maximum 400 caractères) par carte. '
                'Pas de référence à une image absente. N’invente aucune règle. Texte brut. '
                + SVG_INSTRUCTIONS +
                ' Pour ce paquet de cartes, utilise le schéma FlashcardPack : chaque carte a question, answer, page, svg, svg_side. '
                'La page de la carte justifie aussi son dessin ; pas de champ illustration_pages. '
                'Le recto est un exercice autonome : aucune référence de page, citation, lien PDF '
                'ni consigne de consulter le mémento dans question ou dans le SVG du recto. '
                'Garde la référence uniquement dans page ; elle sera affichée après révélation. '
                'Si la source décrit suffisamment les symboles, inclus au moins une carte avec SVG au recto '
                'et une question courte comme « Que représente ce symbole ? ». '
                'svg_side=front pour identifier un symbole : aucun nom de réponse ni légende révélatrice dans le SVG, ses titres ou descriptions. '
                'svg_side=back pour rappeler un symbole : le dessin est caché jusqu’à la révélation. '
                'Si les attributs visuels sont insuffisants, svg vide et question textuelle autonome.',
                f'Notion : {kc.title}. Crée 3 à 5 cartes sur les définitions, règles et distinctions. '
                f'Chaque page doit appartenir à {pages}.\n{source}', FlashcardPack, ctx)
            cards = []
            seen = set()
            for card in out.cards:
                question, answer = card.question.strip(), card.answer.strip()
                if question and answer and card.page in pages and question.casefold() not in seen:
                    svg = ''
                    if card.svg:
                        try:
                            svg = validate_svg(card.svg)
                        except (ValueError, SyntaxError) as exc:
                            self._record_event(sess, ctx, {'event': 'flashcard_svg_rejected',
                                'kc_id': kc.id, 'page': card.page, 'reason': str(exc)[:200]})
                            # A question asking to identify a missing drawing is unusable.
                            if card.svg_side == 'front':
                                continue
                    cards.append(dict(question=question, answer=answer, page=card.page,
                                      svg=svg, svg_side=card.svg_side))
                    seen.add(question.casefold())
            if len(cards) < 3:
                raise RuntimeError('Cartes de révision inexploitables : contenu ou sources insuffisants.')
            deck = dict(id=uuid.uuid4().hex[:12], kc_id=kc.id, cards=cards[:5], index=0,
                        revealed=False, review_indices=[], queue=list(range(len(cards[:5]))),
                        ratings=[], latest_ratings={})
            sess.flashcard_deck = deck
            self._record_event(sess, ctx, {'event': 'flashcards_generated', 'kc_id': kc.id,
                'deck_id': deck['id'], 'cards': deck['cards'], 'evidence_role': 'learning_material_only'})
        if deck['index'] >= len(deck['queue']):
            return [{'type': 'widget', 'title': 'Cartes de revision', 'flashcard': True,
                     'widget': Card(size='md', padding=4, children=[Col(align='end', gap=3, children=[
                         Title(value='Revision terminee', textAlign='end'),
                         Text(value=f"{len(deck['review_indices'])} carte(s) à revoir. Le quiz permet de valider la notion.", textAlign='end'),
                         Row(width='100%', justify='end', wrap='wrap', gap=2,
                             children=[W._button(label, command) for label, command in self._next_actions(sess)])])])}]
        card_id = deck['queue'][deck['index']]
        card = deck['cards'][card_id]
        suffix = f"{deck['id']} {deck['index']}"
        buttons = ([('A revoir', 'fiche encore ' + suffix), ('Difficile', 'fiche difficile ' + suffix), ('Je sais', 'fiche sais ' + suffix)]
                   if deck['revealed'] else [('Voir la réponse', 'fiche reveler ' + suffix)])
        body = card['question']
        illustration = []
        if card.get('svg') and (deck['revealed'] or card.get('svg_side', 'back') == 'front'):
            try:
                svg = validate_svg(card['svg'])
                illustration = [inline_illustration_widget(svg)]
            except (ValueError, SyntaxError):
                body += '\n\nSchéma indisponible. Consultez la source après révélation.'
        children = [Caption(value=kc.title)]
        for block in illustration:
            children.extend(block['widget'].children)
        if deck['revealed']:
            children.extend([Divider(), Text(value=card['answer']),
                Caption(value=f"Source : mémento GOC, p. {card['page']}."),
                Button(label='Consulter la page du PDF', onClickAction=ActionConfig(
                    type='report.open', handler='client', payload={'url': self.doc.pdf_url(card['page'])}))])
        else:
            children.append(Title(value=body, size='md'))
        children.append(Row(justify='center', wrap='wrap', gap=2, children=[W._button(label, command, primary=i == 0)
                                      for i, (label, command) in enumerate(buttons)]))
        known = sum(r == 'know' for r in deck['latest_ratings'].values())
        children.append(Caption(value=f"{card_id + 1} / {len(deck['cards'])} · {known} carte(s) déclarée(s) connue(s) · {len(deck['queue']) - deck['index']} passage(s) restant(s)"))
        children.append(Caption(value='Auto-évaluation de révision, sans validation de notion.'))
        return [{'type': 'widget', 'title': 'Cartes de revision', 'flashcard': True,
                 'widget': Card(size='md', padding=4, children=[Col(align='center', gap=3, children=children)])}]
