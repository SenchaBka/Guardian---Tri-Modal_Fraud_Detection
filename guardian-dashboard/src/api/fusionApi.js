const BASE = '/api/v1/fusion'
const AUDIT_BASE = '/api/v1/audit'

export async function checkHealth() {
  const res = await fetch(`${BASE}/health`, { signal: AbortSignal.timeout(2500) })
  if (!res.ok) throw new Error('Unhealthy')
  return res.json()
}

export async function evaluate(payload) {
  const res = await fetch(`${BASE}/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(10000),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export async function getAuditList() {
  const res = await fetch(AUDIT_BASE)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

const RISK_COUNTRIES = ['NG','RU','CN','BR','IN','PK','KE','GH']
const HIGH_RISK_CATS = ['gambling','crypto','wire_transfer','cash_advance','forex']
const TYPO_PATTERNS  = [/amaz[o0]n/i,/paypa[l1]/i,/app[l1]e/i,/g[o0]{2}gle/i,/micr[o0]soft/i,/[a4]mazon/i]
const RISK_WORDS     = ['urgent','wire','bitcoin','crypto','prize','winner','lottery','transfer','offshore']

export function simulateNumerical(txn) {
  const amt  = parseFloat(txn.amount) || 0
  const ctry = (txn.country || 'US').toUpperCase()
  const cat  = (txn.merchant_category || '').toLowerCase()

  const amtAnomaly  = amt > 5000 ? Math.min(0.95, 0.5 + (amt-5000)/20000) : Math.max(0.08, Math.min(0.35, amt/10000))
  const geoRisk     = RISK_COUNTRIES.includes(ctry) ? 0.72 : 0.11
  const patternRisk = HIGH_RISK_CATS.includes(cat)  ? 0.79 : 0.17
  const velRisk     = 0.24

  const score = +(amtAnomaly*0.35 + velRisk*0.25 + patternRisk*0.25 + geoRisk*0.15).toFixed(4)
  return {
    fraud_probability: score,
    confidence: 0.91,
    signals: { amount_anomaly: +amtAnomaly.toFixed(4), velocity_risk: +velRisk.toFixed(4), pattern_deviation: +patternRisk.toFixed(4), geo_risk: +geoRisk.toFixed(4) },
    flags: [
      amt > 5000 && { label: 'Large transaction', detail: `$${amt.toLocaleString()} — ${(amtAnomaly*100).toFixed(0)}% anomaly score`, severity: amtAnomaly > 0.7 ? 'high' : 'med' },
      RISK_COUNTRIES.includes(ctry) && { label: 'High-risk country', detail: ctry, severity: 'high' },
      HIGH_RISK_CATS.includes(cat)  && { label: 'High-risk category', detail: cat, severity: 'high' },
    ].filter(Boolean),
  }
}

export function simulateNLP(text, narrative) {
  const all = `${text} ${narrative}`.toLowerCase()
  let typoRisk = 0.08, semanticRisk = 0.15, entityRisk = 0.12
  const typoMatches = []
  const riskWordMatches = []

  for (const pat of TYPO_PATTERNS) {
    if (pat.test(text)) { typoRisk = Math.max(typoRisk, 0.88); typoMatches.push(text.match(pat)?.[0]) }
  }
  if (/[o0]/.test(text) && typoRisk < 0.5) { typoRisk = Math.max(typoRisk, 0.52) }

  for (const word of RISK_WORDS) {
    if (all.includes(word)) { semanticRisk = Math.max(semanticRisk, 0.74); riskWordMatches.push(word) }
  }

  const score = +(semanticRisk*0.35 + typoRisk*0.45 + entityRisk*0.20).toFixed(4)
  return {
    fraud_probability: score,
    confidence: 0.85,
    signals: { semantic_risk: +semanticRisk.toFixed(4), typosquatting_risk: +typoRisk.toFixed(4), entity_inconsistency: +entityRisk.toFixed(4) },
    flags: [
      typoMatches.length   && { label: 'Typosquatting detected', detail: `"${typoMatches[0]}" — character substitution pattern`, severity: 'high' },
      riskWordMatches.length && { label: 'High-risk language', detail: riskWordMatches.slice(0,3).join(', '), severity: 'high' },
    ].filter(Boolean),
    highlighted_text: text,
    typo_matches: typoMatches,
    risk_words: riskWordMatches,
  }
}

export function simulateVoice(fileName, isDeepfake) {
  const deepfakeProb = isDeepfake
    ? +(0.88 + Math.random()*0.09).toFixed(4)
    : +(0.04 + Math.random()*0.08).toFixed(4)
  const bonafide = +(1 - deepfakeProb).toFixed(4)
  const score = deepfakeProb
  return {
    fraud_probability: score,
    confidence: 0.92,
    is_deepfake: deepfakeProb > 0.5,
    bonafide_prob: bonafide,
    deepfake_probability: deepfakeProb,
    signals: { deepfake_probability: deepfakeProb, voice_stress_indicator: +(0.1 + Math.random()*0.3).toFixed(4), speaker_consistency: bonafide },
    flags: [
      deepfakeProb > 0.5 && { label: 'Deepfake audio detected', detail: `${(deepfakeProb*100).toFixed(1)}% synthetic probability`, severity: 'high' },
      deepfakeProb < 0.2 && { label: 'Genuine voice confirmed', detail: `${(bonafide*100).toFixed(1)}% bonafide probability`, severity: 'ok' },
    ].filter(Boolean),
    model: 'microsoft/wavlm-base',
    filename: fileName,
  }
}

export function simulateFusion(numResult, nlpResult, voiceResult) {
  // Guard: if numResult looks like raw form fields (no fraud_probability),
  // run simulateNumerical on it first. Handles version mismatch gracefully.
  if (numResult && !numResult.fraud_probability) {
    numResult = simulateNumerical(numResult)
  }
  if (!numResult) return null

  const nlp   = nlpResult   != null
  const voice = voiceResult != null
  const wBase = { numerical: 0.45, nlp: nlp ? 0.35 : 0, voice: voice ? 0.20 : 0 }
  const total = wBase.numerical + wBase.nlp + wBase.voice
  const w     = { numerical: wBase.numerical/total, nlp: wBase.nlp/total, voice: wBase.voice/total }

  let score = w.numerical * numResult.fraud_probability
  if (nlp)   score += w.nlp   * nlpResult.fraud_probability
  if (voice) score += w.voice * voiceResult.fraud_probability
  score = Math.max(0, Math.min(1, score))

  const penalty = (!nlp && !voice) ? 0.40 : (!nlp ? 0.20 : !voice ? 0.15 : 0.0)
  const baseConf = (numResult.confidence + (nlp ? nlpResult.confidence : 0) + (voice ? voiceResult.confidence : 0)) / (1 + (nlp?1:0) + (voice?1:0))
  const conf = +(baseConf * (1 - penalty)).toFixed(4)
  const mode = (!nlp && !voice) ? 'fallback' : (nlp && voice ? 'full' : 'partial')
  const riskLevel = score <= 0.25 ? 'low' : score <= 0.50 ? 'medium' : score <= 0.75 ? 'high' : 'critical'

  let decision
  if (mode === 'fallback') decision = score >= 0.60 ? 'block' : score > 0.25 ? 'review' : 'approve'
  else                     decision = score >= 0.71 ? 'block' : score > 0.30 ? 'review' : 'approve'

  const missing = [...(!nlp?['nlp']:[]),...(!voice?['voice']:[])];
  const reqId = 'req-' + Math.random().toString(16).slice(2,10)+'-'+Date.now().toString(16)

  const sig = numResult.signals || {}
  const topFactors = []
  if ((sig.amount_anomaly    || 0) > 0.4) topFactors.push({ feature:'Transaction Amount Anomaly',    impact:+(w.numerical*(sig.amount_anomaly   ||0)*0.9).toFixed(4), direction:'positive' })
  if ((sig.geo_risk          || 0) > 0.4) topFactors.push({ feature:'Geographic Risk',               impact:+(w.numerical*(sig.geo_risk         ||0)*0.8).toFixed(4), direction:'positive' })
  if ((sig.pattern_deviation || 0) > 0.4) topFactors.push({ feature:'Behavioral Pattern Deviation',  impact:+(w.numerical*(sig.pattern_deviation||0)*0.7).toFixed(4), direction:'positive' })
  if (nlp && nlpResult.signals.typosquatting_risk > 0.3) topFactors.push({ feature:'Merchant Typosquatting Risk', impact:+(w.nlp*nlpResult.signals.typosquatting_risk*0.9).toFixed(4), direction:'positive' })
  if (nlp && nlpResult.signals.semantic_risk > 0.3)      topFactors.push({ feature:'Semantic Risk in Description', impact:+(w.nlp*nlpResult.signals.semantic_risk*0.8).toFixed(4), direction:'positive' })
  if (voice && voiceResult.is_deepfake)                  topFactors.push({ feature:'Deepfake Voice Detected', impact:+(w.voice*voiceResult.deepfake_probability*0.95).toFixed(4), direction:'positive' })
  topFactors.sort((a,b) => b.impact - a.impact)

  return {
    transaction_id: 'TXN-'+Date.now().toString(36).toUpperCase(),
    fraud_score: +score.toFixed(4),
    decision,
    confidence: conf,
    risk_level: riskLevel,
    ensemble_mode: mode,
    model_version: '2.0.0',
    modality_scores: {
      numerical: { score: numResult.fraud_probability,           status:'ok',          weight:w.numerical, used:true },
      nlp:       { score: nlp   ? nlpResult.fraud_probability   : null, status:nlp?'ok':'unavailable',   weight:w.nlp,   used:nlp   },
      voice:     { score: voice ? voiceResult.fraud_probability  : null, status:voice?'ok':'unavailable', weight:w.voice, used:voice },
    },
    confidence_adjustment: { base_confidence:+baseConf.toFixed(4), availability_penalty:penalty, final_confidence:conf },
    explanation: {
      top_factors: topFactors,
      shap_values: {
        numerical_contribution: +(w.numerical*(numResult.fraud_probability-0.02)).toFixed(4),
        nlp_contribution:   nlp   ? +(w.nlp*(nlpResult.fraud_probability-0.02)).toFixed(4)   : null,
        voice_contribution: voice ? +(w.voice*(voiceResult.fraud_probability-0.02)).toFixed(4) : null,
      },
      narrative: `${score>=0.71?'High':score>=0.31?'Moderate':'Low'} fraud risk (${score.toFixed(2)}) via ${mode} ensemble. ${topFactors[0]?.feature||'No dominant factor'} was the primary signal.`,
      missing_modality_note: missing.length ? `${missing.map(m=>m.toUpperCase()).join(' and ')} stream(s) unavailable for this transaction type.` : null,
    },
    audit: {
      request_id: reqId,
      processing_time_ms: +(Math.random()*40+12).toFixed(1),
      modalities_requested: ['numerical',...(nlp?['nlp']:[]),...(voice?['voice']:[])],
      modalities_used:      ['numerical',...(nlp?['nlp']:[]),...(voice?['voice']:[])],
      modalities_failed: [],
    },
  }
}