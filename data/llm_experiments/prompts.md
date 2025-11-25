# LLM Drift Generation Prompts

Use these prompts in claude.ai

---

## Prompt 1: Length Drift (3x Longer)

You are helping test a mental health text classification system's drift detection.

TASK: Transform these mental health texts to be 3x longer while keeping the same core message and mental health category.

INSTRUCTIONS:
- Maintain the emotional tone and mental health state
- Add realistic elaboration (more details, context, feelings)
- Keep it authentic to how real people describe mental health
- DO NOT change the underlying mental health category
- Make the text approximately 3x the original length

Original texts:
[PASTE TEXTS HERE]

Respond with ONLY the transformed texts, numbered 1-10, no explanations.

---

## Prompt 2: Gen Z Slang

You are helping test a mental health text classification system's drift detection.

TASK: Transform these mental health texts to use heavy Gen Z slang and informal language.

INSTRUCTIONS:
- Use Gen Z terms: "down bad", "no cap", "fr fr", "lowkey", "highkey", "hella", "deadass"
- Replace formal language with casual equivalents
- Keep the same mental health state and severity
- Make it sound like a young person texting
- Maintain emotional authenticity
- DO NOT change the underlying mental health category

Original texts:
[PASTE TEXTS HERE]

Respond with ONLY the transformed texts, numbered 1-10, no explanations.

---

## Prompt 3: Formality Shift (Casual → Clinical)

You are helping test a mental health text classification system's drift detection.

TASK: Transform these texts to clinical, formal language as if written by a healthcare professional.

INSTRUCTIONS:
- Use clinical terms: experiencing, symptoms, difficulty, unable to, significant, substantial
- Change to third-person or formal first-person
- Remove casual language and contractions
- Keep the same mental health state and severity
- Maintain clinical accuracy

Original texts:
[PASTE TEXTS HERE]

Respond with ONLY the transformed texts, numbered 1-10, no explanations.

---

## Prompt 4: Adversarial (Subtle Changes)

You are helping test a mental health text classification system's drift detection.

TASK: Transform these texts in subtle ways that might evade drift detection while still changing the text meaningfully.

IDEAS TO TRY:
- Synonym replacement (sad → melancholy, anxious → apprehensive)
- Sentence restructuring (same meaning, different structure)
- Adding filler words or phrases
- Changing specific examples while keeping general themes

CONSTRAINTS:
- Keep the same mental health category
- Keep similar length (±10%)
- Make changes that feel natural
- Try to be subtle but meaningful

Original texts:
[PASTE TEXTS HERE]

Respond with ONLY the transformed texts, numbered 1-10, no explanations.

---

## Prompt 5: Multi-Drift (Combined)

You are helping test a mental health text classification system's drift detection.

TASK: Transform these texts with MULTIPLE changes at once:
1. Make them 2-3x longer
2. Use Gen Z slang
3. Add typos and informal punctuation (!!!, ..., emojis)
4. Change formality to very casual

INSTRUCTIONS:
- Combine multiple drift types
- Keep the same mental health category
- Make it feel like a different population is using the system
- Maintain emotional authenticity

Original texts:
[PASTE TEXTS HERE]

Respond with ONLY the transformed texts, numbered 1-10, no explanations.
