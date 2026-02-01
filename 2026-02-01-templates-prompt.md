# Output Templates Review Request

I'm building a RAG (Retrieval-Augmented Generation) research system for exploring a corpus of ~1,600 podcast transcripts and documents about UFOs, consciousness, ancient civilizations, and paranormal phenomena. The system retrieves relevant chunks from the corpus and synthesizes answers.

I want feedback on my output templates. Are they too lengthy? What sections are essential vs. optional? What alternatives would work better?

---

## Context: How the System Works

1. User asks a question like "What do people say about Tall Whites?"
2. System retrieves ~30-50 relevant text chunks from the corpus
3. An LLM synthesizes an answer using ONLY the retrieved content (no external knowledge)
4. Output uses one of the templates below

---

## Current Output Templates

### Template A: Standard Query Response

Used for most queries. System prompts the LLM to produce:

```markdown
## Direct Answer

What does the evidence suggest? Be specific.¹ Grade evidence strength:
- Strong: multiple independent sources agree
- Moderate: some support but limited
- Weak: single source or contradictory

## Connections

List significant connections between sources:

- **Agreements**: Where sources confirm each other
- **Conflicts**: Where sources disagree (note which is more credible)
- **Patterns**: Recurring names, places, or themes across sources

## Rabbit Holes

Threads worth exploring further:
- Topic — why interesting — which sources mention it

---

## Sources

¹ filename.md
² another-file.md
```

**Typical output length**: 400-800 words

---

### Template B: Deep Research Mode

Used for complex multi-agent research queries (spawns parallel search agents). Much more detailed:

```markdown
### Direct Answer
What does the evidence suggest? Be specific and cite sources.
Grade evidence strength:
- **High**: Multiple independent sources agree
- **Med**: Some support, but limited sources
- **Low**: Single source or contradictory evidence

When listing claims in tables, ALWAYS sort by strength: High first, then Med, then Low.

### Connections
Trace how different sources connect:

- **Agreements**: Same claim from independent sources (high value)
- **Conflicts**: Contradicting claims (flag for investigation)
- **Sequences**: Temporal or causal relationships
- **Same subject**: Same person/place/event in different contexts

Use abbreviations: High/Med/Low for strength ratings to fit table columns.
Prioritize agreements and conflicts - these are the most valuable signals.

### Convergence
Where do different perspectives agree? What patterns emerge?

### Tensions
Where do perspectives conflict?
- What specifically do they disagree on?
- Possible explanations: different time periods? biases? incomplete info?
- Which source is more credible, and why?

### Unexpected Connections
What non-obvious links did you find? These serendipitous discoveries are valuable.

### Confidence Assessment
Overall confidence in conclusions: High/Medium/Low
What would increase confidence? What evidence is missing?

### Rabbit Holes
What specific threads deserve deeper exploration? Be concrete.
Format: "[Topic] - Why interesting - Which sources mention it"
```

**Typical output length**: 1,000-2,500 words

---

### Template C: Conversational Chat

Used in chat mode. Minimal structure:

```
You are a knowledgeable research assistant...

## Conversation Style

- Be natural and conversational, like a knowledgeable friend discussing research
- Respond directly to what was asked - don't dump everything you know
- Keep responses focused and appropriately sized for the question
- If the user wants more detail, they'll ask
- Remember previous turns - don't repeat what you've already said
- Use markdown sparingly (headers only for longer responses)
```

**Typical output length**: 100-400 words (varies by question)

---

## Questions for Review

1. **Are these templates too verbose?** The user feels they're too lengthy.

2. **Which sections are essential?**
   - Direct Answer: Required (the core output)
   - Connections: Useful but could be optional?
   - Rabbit Holes: Nice to have but adds length
   - Sources: Required for citations
   - Convergence/Tensions: Overlap with Connections?

3. **Better alternatives?**
   - Shorter default template, with `/detailed` flag for full analysis?
   - Progressive disclosure (summary first, expandable sections)?
   - Different templates for different query complexities?

4. **What's actually useful in research workflows?**
   - Evidence strength grading: essential or noise?
   - "Rabbit Holes" suggestions: helpful or padding?
   - Distinguishing Agreements vs Conflicts vs Patterns: overkill?

5. **Suggested simplified template?**
   Please propose a leaner alternative if you think the current ones are bloated.

---

## Design Constraints

- Output must cite sources (this is a RAG system)
- Output must acknowledge uncertainty (corpus may not cover topic)
- Output goes to terminal (glow markdown renderer) or web UI
- Users range from casual explorers to serious researchers
