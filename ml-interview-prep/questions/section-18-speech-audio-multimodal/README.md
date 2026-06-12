<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 17 — Fine-Tuning, PEFT & Adaptation](../section-17-fine-tuning-peft-adaptation/README.md) &nbsp;|&nbsp; [Section 19 — Prompt Engineering, ICL & Steering ➡️](../section-19-prompt-engineering-icl-steering/README.md)

</div>

---

# Section 18 · Speech, Audio, and Multimodal

<div align="center">

![Questions](https://img.shields.io/badge/questions-20-1f4fa3)
![Levels](https://img.shields.io/badge/levels-Basic→Intermediate→Advanced→Applied-0e8a6e)
![Answered](https://img.shields.io/badge/answered-0%2F20-c77a12)
![Scaffolded](https://img.shields.io/badge/scaffolded-20%2F20-6c4fe0)

</div>

Speech and audio are the next frontier for unified models. This section covers **Whisper and ASR fundamentals**, **TTS architectures** (Tacotron, VITS), **audio tokenization**, **unified speech-text models**, and the frontier of **real-time duplex conversation** in systems like Gemini Live and GPT-4o.

> **Legend** &nbsp; ✅ full answer published &nbsp;·&nbsp; 📝 scaffolded (awaiting write-up)

---

### 🟢 Basic

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q18‑01 | How does Whisper work? What training data and objective made it broadly multilingual? | Whisper · ASR · multilingual · weakly-supervised | 📝 |
| Q18‑02 | Compare CTC, attention-encoder-decoder, and RNN-T for ASR. What does each sacrifice? | CTC · attention-decoder · RNN-T · ASR | 📝 |
| Q18‑03 | What is a mel spectrogram? Why does ASR operate in this domain rather than raw waveform? | mel-spectrogram · STFT · frequency · ASR | 📝 |
| Q18‑04 | How does Tacotron 2 / VITS work? Walk through the key components from text to waveform. | Tacotron · VITS · TTS · vocoder | 📝 |
| Q18‑05 | What is the difference between speaker-dependent and speaker-independent TTS? What enables zero-shot voice cloning? | speaker-dependent · zero-shot · voice-cloning · TTS | 📝 |

---

### 🟡 Intermediate

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q18‑06 | Explain audio tokenization (EnCodec, SoundStream). How do discrete audio tokens enable language-model-style generation? | EnCodec · SoundStream · audio-tokenization · codebook | 📝 |
| Q18‑07 | Walk through AudioLM and MusicGen. How do they use hierarchical codebook tokens for unconditional and conditional audio generation? | AudioLM · MusicGen · hierarchical · conditional | 📝 |
| Q18‑08 | Discuss unified speech-text models. How are speech tokens interleaved with text tokens in models like SpeechGPT and LLaSA? | speech-text · unified · interleaved-tokens · SpeechGPT | 📝 |
| Q18‑09 | What are the latency requirements for low-latency duplex speech interaction? What component dominates latency? | latency · duplex · ASR · TTS · streaming | 📝 |
| Q18‑10 | Compare Gemini Native Audio, GPT-4o native speech, and cascade (ASR→LLM→TTS) approaches. What does end-to-end training buy? | Gemini · GPT-4o · cascade · end-to-end · comparison | 📝 |

---

### 🔴 Advanced

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q18‑11 | Walk through the design of a real-time speech-to-speech system with <300ms end-to-end latency. What are the architectural choices? | real-time · speech-to-speech · latency · streaming | 📝 |
| Q18‑12 | What metrics should you use to evaluate speech generation beyond WER? Discuss MOS, UTMOS, NISQA, naturalness vs intelligibility. | MOS · UTMOS · NISQA · evaluation · TTS | 📝 |
| Q18‑13 | Critically evaluate whether unified speech-text models are better than specialized ASR+TTS pipelines. What does the empirical evidence say so far? | unified-vs-specialized · speech · empirical · tradeoffs | 📝 |
| Q18‑14 | Discuss the design of a joint multimodal embedding space for speech, text, and images. What are the alignment challenges? | multimodal-embedding · speech-text-image · alignment | 📝 |
| Q18‑15 | Discuss the role of speech in agentic systems. How do Siri/Google Assistant/Alexa differ from emerging LLM-native voice agents? | speech-agents · Siri · Google-Assistant · LLM-native | 📝 |

---

### 🔵 Applied / System Design

| # | Question | Tags | Status |
|---|----------|------|:------:|
| Q18‑16 | Your ASR model has 5% WER on a benchmark but 18% WER in production on customer calls. Walk through your investigation. | WER-gap · production · benchmark · domain-shift | 📝 |
| Q18‑17 | Walk through how you'd build a real-time voice assistant that can interrupt and be interrupted naturally. | interruption · turn-taking · streaming · duplex | 📝 |
| Q18‑18 | Your TTS model produces unnatural prosody on questions but not statements. Walk through your investigation and fix. | prosody · questions · TTS · debugging | 📝 |
| Q18‑19 | Walk through how you'd adapt a Whisper model to a low-resource language with 100 hours of transcribed audio. | low-resource · Whisper · adaptation · 100-hours | 📝 |
| Q18‑20 | Walk through how you'd evaluate whether a new end-to-end speech model is better than your existing ASR+TTS cascade. | end-to-end-eval · cascade · tradeoffs · A/B-testing | 📝 |

---

## Why these questions matter

Speech and audio capabilities are increasingly core to production AI products. These questions test whether a candidate understands the **fundamental ASR/TTS tradeoffs, audio tokenization, and the emerging unified model landscape**.

| Theme | Key questions |
|-------|--------------|
| **ASR fundamentals** | Q18‑01, Q18‑02, Q18‑03 — Whisper, CTC, mel spectrogram |
| **TTS & generation** | Q18‑04, Q18‑05, Q18‑07 — Tacotron, voice cloning, AudioLM |
| **Unified models** | Q18‑08, Q18‑10, Q18‑13 — speech-text, Gemini, GPT-4o |
| **Production & latency** | Q18‑09, Q18‑11, Q18‑16 — duplex, streaming, WER gap |

---

## Reading order suggestions

**New to speech AI?** Q18‑01 → Q18‑03 → Q18‑04 → Q18‑06 → Q18‑10

**Interview in 48 hours?** Q18‑02 (ASR tradeoffs) · Q18‑06 (audio tokenization) · Q18‑10 (Gemini/GPT-4o) · Q18‑11 (latency design) · Q18‑13 (unified vs cascade)

**Full track:** Q18‑01 → Q18‑05 → Q18‑06 → Q18‑08 → Q18‑10 → Q18‑11 → Q18‑13

---

<div align="center">

[🏠 Home](../../README.md) &nbsp;|&nbsp; [⬅️ Section 17 — Fine-Tuning, PEFT & Adaptation](../section-17-fine-tuning-peft-adaptation/README.md) &nbsp;|&nbsp; [Section 19 — Prompt Engineering, ICL & Steering ➡️](../section-19-prompt-engineering-icl-steering/README.md)

<sub>No answer yet? Copy <a href="../../_TEMPLATE.md"><code>_TEMPLATE.md</code></a>, fill the blanks, add figures under <code>assets/s18qNN/</code>, then flip the status to ✅ here. See <a href="../../CONTRIBUTING.md">CONTRIBUTING</a>.</sub>

</div>
