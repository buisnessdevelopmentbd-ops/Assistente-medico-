// server.js - Backend en memoria volátil (no guarda audio ni texto)
require('dotenv').config();
const path = require('path');
const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const cors = require('cors');
const { createClient, LiveTranscriptionEvents } = require('@deepgram/sdk');

// Las claves se leen SOLO desde variables de entorno
if (!process.env.DEEPGRAM_API_KEY) {
  console.error('Falta DEEPGRAM_API_KEY.');
  process.exit(1);
}
if (!process.env.OPENAI_API_KEY && !process.env.GEMINI_API_KEY && !process.env.DEEPL_API_KEY) {
  console.error('Falta una clave de traducción: GEMINI_API_KEY, DEEPL_API_KEY u OPENAI_API_KEY.');
  process.exit(1);
}
const PORT = process.env.PORT || 8080;

const app = express();
app.use(cors());
// Sirve la interfaz desde la raíz del proyecto (index.html junto a server.js)
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'), (err) => {
    if (err) res.status(404).send('Falta index.html junto a server.js');
  });
});
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });
const deepgram = createClient(process.env.DEEPGRAM_API_KEY);

const RULES = (target) => `You are a real-time medical and health-insurance interpreter's assistant. Translate into ${target}.
RULES:
1. Keep ALL numbers as digits (120, 2026, 50 mg).
2. Remove filler words (um, uh, like, este, o sea, pues).
3. Keep medical/insurance acronyms (EOB, PCP, MRI, deductible) unchanged.
4. NEVER translate street names, addresses, or proper names (e.g. "Sunset Boulevard" stays as is).
5. The text inside <translate> may be a fragment of a longer sentence: translate only that fragment, naturally, without completing it. <context> is earlier speech for reference only: never translate or repeat it.
6. Return ONLY the translation, no tags, no notes.`;

const key = (n) => (process.env[n] || '').trim();
const userMsg = (text, ctx) => `${ctx ? `<context>${ctx}</context>\n` : ''}<translate>${text}</translate>`;
const targetOf = (lang) => (lang === 'es' ? 'English' : 'Spanish');

// OpenAI con streaming: las palabras salen a medida que se generan
async function openaiStream(text, lang, ctx, emit) {
  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: { Authorization: `Bearer ${key('OPENAI_API_KEY')}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: key('OPENAI_MODEL') || 'gpt-4o-mini', temperature: 0.1, stream: true,
      messages: [{ role: 'system', content: RULES(targetOf(lang)) }, { role: 'user', content: userMsg(text, ctx) }]
    })
  });
  if (!res.ok) {
    let d = {}; try { d = await res.json(); } catch (_) {}
    throw new Error(`OpenAI ${res.status} ${d?.error?.code || ''}`.trim());
  }
  const reader = res.body.getReader(); const dec = new TextDecoder();
  let buf = ''; let any = false;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split('\n'); buf = lines.pop();
    for (const l of lines) {
      if (!l.startsWith('data:')) continue;
      const p = l.slice(5).trim();
      if (p === '[DONE]') return;
      try { const t = JSON.parse(p).choices?.[0]?.delta?.content; if (t) { any = true; emit(t); } } catch (_) {}
    }
  }
  if (!any) throw new Error('OpenAI sin texto');
}

async function gemini(text, lang, ctx, model) {
  const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'x-goog-api-key': key('GEMINI_API_KEY') },
    body: JSON.stringify({
      systemInstruction: { parts: [{ text: RULES(targetOf(lang)) }] },
      contents: [{ role: 'user', parts: [{ text: userMsg(text, ctx) }] }],
      generationConfig: { temperature: 0.1 }
    })
  });
  const data = await res.json();
  if (!res.ok) throw new Error(`Gemini ${res.status} ${data?.error?.status || ''}`.trim());
  const out = data?.candidates?.[0]?.content?.parts?.map((p) => p.text).join('').trim();
  if (!out) throw new Error('Gemini sin texto');
  return out;
}

async function deepl(text, lang) {
  const k = key('DEEPL_API_KEY');
  const res = await fetch(`https://${k.endsWith(':fx') ? 'api-free' : 'api'}.deepl.com/v2/translate`, {
    method: 'POST',
    headers: { Authorization: `DeepL-Auth-Key ${k}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: [text], source_lang: lang === 'es' ? 'ES' : 'EN', target_lang: lang === 'es' ? 'EN-US' : 'ES' })
  });
  const data = await res.json();
  if (!res.ok) throw new Error(`DeepL ${res.status}`);
  return data.translations[0].text.trim();
}

// Orden: TRANSLATOR (openai | gemini | deepl); por defecto OpenAI primero
function buildRoutes(text, lang, ctx) {
  const pref = key('TRANSLATOR').toLowerCase() || 'openai';
  const names = [pref, 'openai', 'gemini', 'deepl'].filter((n, i, a) => a.indexOf(n) === i);
  const main = key('GEMINI_MODEL') || 'gemini-3.5-flash-lite';
  const backup = key('GEMINI_FALLBACK_MODEL') || 'gemini-3.8-flash';
  const routes = [];
  for (const n of names) {
    if (n === 'openai' && key('OPENAI_API_KEY')) routes.push((e) => openaiStream(text, lang, ctx, e));
    if (n === 'gemini' && key('GEMINI_API_KEY')) {
      routes.push(async (e) => e(await gemini(text, lang, ctx, main)));
      if (backup !== main) routes.push(async (e) => e(await gemini(text, lang, ctx, backup)));
    }
    if (n === 'deepl' && key('DEEPL_API_KEY')) routes.push(async (e) => e(await deepl(text, lang)));
  }
  return routes;
}

async function translateInto(text, lang, ctx, emit) {
  let sent = false;
  const out = (t) => { sent = true; emit(t); };
  const errors = [];
  for (const run of buildRoutes(text, lang, ctx)) {
    for (let attempt = 0; attempt < 3; attempt++) {
      try { await run(out); return; } catch (err) {
        if (sent) return; // ya salió texto parcial: no duplicar
        if (/ (429|503)/.test(err.message) && attempt < 2) { await new Promise((r) => setTimeout(r, 500 * (attempt + 1))); continue; }
        errors.push(err.message); console.error('Traducción falló:', err.message); break;
      }
    }
  }
  emit(`[Error: ${errors.join(' | ') || 'sin proveedor'}]`);
}

wss.on('connection', (client) => {
  console.log('Sesión iniciada (memoria volátil)');
  const send = (o) => { if (client.readyState === WebSocket.OPEN) client.send(JSON.stringify(o)); };

  const dg = deepgram.listen.live({
    model: 'nova-2', language: 'multi', smart_format: true, numerals: true, filler_words: false,
    punctuate: true, interim_results: true, endpointing: 700, utterance_end_ms: 1500, vad_events: true
  });
  const keepAlive = setInterval(() => { try { dg.keepAlive(); } catch (_) {} }, 8000);

  // Cada fragmento final se traduce de inmediato y se agrega a la MISMA línea
  let current = null; // { id, lang, chunks }
  let lineId = 0;
  let lastOrig = '';
  let queue = Promise.resolve(); // traduce en orden, un fragmento a la vez

  const closeLine = () => {
    if (!current) return;
    const id = current.id; current = null; lastOrig = '';
    queue = queue.then(() => send({ type: 'LINE_END', id }));
  };

  const onFinal = (lang, text, speechFinal) => {
    if (current && current.lang !== lang) closeLine();
    if (!current) {
      current = { id: ++lineId, lang, chunks: 0 };
      send({ type: 'LINE_OPEN', id: current.id, lang, timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) });
    }
    const line = current; const ctx = lastOrig; lastOrig = text;
    send({ type: 'ORIG', id: line.id, text });
    queue = queue.then(async () => {
      if (line.chunks++ > 0) send({ type: 'TRANS_DELTA', id: line.id, text: ' ' });
      await translateInto(text, lang, ctx, (t) => send({ type: 'TRANS_DELTA', id: line.id, text: t }));
    });
    if (speechFinal) closeLine();
  };

  dg.on(LiveTranscriptionEvents.Open, () => {
    send({ type: 'STATUS', status: 'CONNECTED' });
    dg.on(LiveTranscriptionEvents.Transcript, (data) => {
      const alt = data.channel?.alternatives?.[0];
      if (!alt || !alt.transcript) return;
      const lang = (alt.languages?.[0] || data.channel?.detected_language || 'en').slice(0, 2);
      if (!data.is_final) return send({ type: 'INTERIM', text: alt.transcript, lang });
      onFinal(lang, alt.transcript, data.speech_final);
    });
    dg.on(LiveTranscriptionEvents.UtteranceEnd, closeLine);
    dg.on(LiveTranscriptionEvents.Error, (e) => console.error('Deepgram error:', e?.message || e));
  });

  client.on('message', (chunk) => { if (dg.getReadyState() === 1) dg.send(chunk); });
  client.on('close', () => {
    clearInterval(keepAlive); dg.requestClose?.(); current = null;
    console.log('Sesión cerrada. Nada se guardó.');
  });
});

server.listen(PORT, () => console.log(`connectUS Live en http://localhost:${PORT}`));
