"""Optional personalized narration; no assessment or mastery updates."""
import base64
import hashlib
import html
import json
import os
from openai import AsyncOpenAI
from pydantic import BaseModel
from chatkit.actions import ActionConfig
from chatkit.widgets import Button, Card, Col, Text
from app.providers import current_provider, run_structured


class PodcastScript(BaseModel):
    transcript: str
    pages: list[int]


async def synthesize(transcript, key):
    async with AsyncOpenAI(api_key=key, max_retries=0, timeout=90) as client:
        response = await client.audio.speech.create(
            model=os.getenv('OPENAI_TTS_MODEL', 'gpt-4o-mini-tts'),
            voice=os.getenv('OPENAI_TTS_VOICE', 'coral'), input=transcript,
            response_format='mp3')
        return response.content


def player_html(title, transcript, audio, links):
    encoded = base64.b64encode(audio).decode('ascii')
    sources = ' · '.join(f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener">Page {page}</a>' for page, url in links)
    return ('<!doctype html><html lang="fr"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<body style="font-family:system-ui;padding:24px;line-height:1.6">'
            f'<h1>{html.escape(title)}</h1><p>Voix générée par IA.</p>'
            f'<audio controls preload="none" style="width:100%" src="data:audio/mpeg;base64,{encoded}"></audio>'
            f'<p>{sources}</p><h2>Transcription</h2><p style="white-space:pre-wrap">'
            f'{html.escape(transcript)}</p></body></html>')


class Podcasts:
    async def _podcast(self, sess, ctx):
        kc = self._kc(sess.current_kc_id)
        if sess.phase == 'waiting_answers':
            return [self._text('Terminez le quiz avant d’écouter une leçon audio.')]
        if not kc or (not sess.diagnostic_done and sess.diagnostic_stage != 'learning'):
            return [self._text('La leçon audio est disponible pendant l’apprentissage d’une notion, après son diagnostic.')]
        pages = self.graph.kc_pages(kc)
        source = self.doc.pages_text(pages, max_chars=10000)
        if not pages or not source.strip():
            return [self._text('Source indisponible pour préparer la leçon audio.')]
        choice = current_provider.get()
        key = (choice.api_key if choice.provider == 'openai' else None) or os.getenv('OPENAI_API_KEY')
        fingerprint = hashlib.sha256(json.dumps([kc.id, source, sess.last_mistakes_summary,
            sess.misconceptions.get(kc.id, []), choice.provider,
            os.getenv('OPENAI_TTS_MODEL', 'gpt-4o-mini-tts'), os.getenv('OPENAI_TTS_VOICE', 'coral')],
            ensure_ascii=False).encode()).hexdigest()
        cached = sess.podcast_cache
        if cached.get('fingerprint') != fingerprint:
            cached = {'fingerprint': fingerprint}
        transcript = cached.get('transcript')
        if not transcript:
            if not key:
                return [self._text('Une clé OpenAI est nécessaire pour la synthèse vocale. Configurez la clé serveur ou sélectionnez OpenAI avec votre clé personnelle.')]
            self._charge_budget(sess)
            script = await run_structured('Podcast-script',
                'Écris une courte leçon orale française, 200 à 350 mots, uniquement à partir des extraits. '
                'Explique les règles puis un exemple et termine par une question de réflexion. '
                'Cible les erreurs observées sans affirmer une misconception certaine. '
                'Ne prétends pas montrer des images. Invite à consulter le PDF pour les symboles. '
                'Texte brut; pages contient seulement les pages utilisées.',
                f'Notion : {kc.title}. Erreurs observées : {sess.last_mistakes_summary}. '
                f'Confusions enregistrées : {sess.misconceptions.get(kc.id, [])}. '
                f'Pages autorisées : {pages}.\n{source}', PodcastScript, ctx)
            transcript = script.transcript.strip()
            if not transcript or len(transcript) > 4000 or not script.pages or not set(script.pages).issubset(pages):
                raise RuntimeError('Leçon audio inexploitable : texte ou références invalides.')
            cached.update(transcript=transcript, pages=sorted(set(script.pages)))
            sess.podcast_cache = cached
            self._record_event(sess, ctx, {'event': 'podcast_script_generated', 'kc_id': kc.id,
                'transcript': transcript, 'source_pages': cached['pages'], 'evidence_role': 'learning_material_only'})
        if not cached.get('audio'):
            if not key:
                return [self._text(transcript), self._text('Clé OpenAI manquante pour la synthèse vocale.')]
            self._charge_budget(sess)
            try:
                audio = await synthesize(transcript, key)
                if not audio or len(audio) > 8_000_000:
                    raise ValueError('Invalid audio size')
            except Exception as exc:
                self._record_event(sess, ctx, {'event': 'podcast_audio_failed', 'kc_id': kc.id,
                    'error_type': type(exc).__name__})
                return [self._text(transcript), self._text('Synthèse vocale indisponible. Le texte est conservé ; réessayez « podcast ».'),
                        self._actions([('Réessayer l’audio', 'podcast')])]
            cached['audio'] = base64.b64encode(audio).decode('ascii')
            sess.podcast_cache = cached
            self._record_event(sess, ctx, {'event': 'podcast_audio_generated', 'kc_id': kc.id,
                'source_pages': cached['pages'], 'evidence_role': 'learning_material_only'})
        document = player_html(kc.title, transcript, base64.b64decode(cached['audio']),
                               [(p, self.doc.pdf_url(p)) for p in cached['pages']])
        widget = Card(children=[Col(children=[Text(value='Leçon audio personnalisée · voix générée par IA'),
            Button(label='Écouter et lire', onClickAction=ActionConfig(type='report.open', handler='client',
                payload={'html': document, 'title': kc.title, 'url': ''}))])])
        self._record_event(sess, ctx, {'event': 'podcast_presented', 'kc_id': kc.id,
            'evidence_role': 'available_to_play_not_proof_of_listening'})
        return [self._text(transcript), {'type': 'widget', 'title': 'Leçon audio', 'widget': widget},
                self._actions(self._next_actions(sess))]
