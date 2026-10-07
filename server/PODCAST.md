# Personalized audio lessons

During learning, click ?couter la le?on or type `podcast` / `le?on audio`.
Finish pending quizzes first. The current KC page excerpts and recorded mistakes
feed a French script; OpenAI speech synthesis narrates it. ?couter et lire opens
an HTML audio player, transcript and PDF references in the existing side viewer.
The player discloses that the voice is AI-generated. No autoplay.

Speech requires OPENAI_API_KEY or a personal key when OpenAI is selected.
A Mistral personal key is never sent to OpenAI. Script generation uses the selected
text provider. Defaults: OPENAI_TTS_MODEL=gpt-4o-mini-tts, OPENAI_TTS_VOICE=coral.
A fresh lesson charges one script generation and one speech generation against the
learner daily budget. Reopening unchanged content reuses the saved audio. Failed
speech retains the script; retry only charges speech. Context changes invalidate
this single-entry cache. Reset clears it with the session.

For this first version MP3 is embedded in the viewer HTML and saved as base64 in
the learner session. This increases session/thread/log size; audio is bounded to
8 MB. Do not expose session storage publicly. The feature adds no public media
endpoint. Transcript/page validation is structural, not expert semantic review.

Events: podcast_script_generated, podcast_audio_generated, podcast_audio_failed,
podcast_presented. Presentation is not proof of listening. No mastery, validated
KC or diagnostic evidence changes occur. The simulation runner does not request
podcasts automatically; its model-call wrapper does not count direct speech calls.
Do not use that wrapper as a speech cost limit if extending simulations to audio.

Offline tests: python -m unittest test_podcast -q (from server).
Live audio generation/playback needs an API-enabled manual check; no paid calls
were made during implementation.

API reference: https://developers.openai.com/api/docs/guides/text-to-speech
