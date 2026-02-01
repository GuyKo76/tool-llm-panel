# Project Brief: Career Discovery Agents
## For Expert LLM Consultation

**Prepared:** 2026-02-01
**Updated:** 2026-02-02
**Focus:** Profile Accuracy & Serendipitous Job Discovery

---

## Consultation Focus

**Primary Goal:** The system's purpose is to create a candidate profile rich enough that a downstream agent can discover non-obvious job matches - jobs the user doesn't even know exist but would genuinely fit them.

**Current Pain Points:**
1. Questioning feels repetitive and predictable
2. Doubt that current approach uncovers latent/hidden characteristics
3. Need deeper psychological/behavioral profiling, not just skills and experience
4. The "serendipity" feels forced rather than genuinely surprising

**What We Need Help With:**
- How to uncover traits users might not articulate about themselves
- Questioning techniques that reveal hidden patterns
- Profile structure that enables genuine serendipitous matching
- How to surface what makes someone unusual in non-obvious ways

---

## Executive Summary

This is a **4-agent career discovery pipeline** with a FastAPI web application:

1. **career-pathfinder** (Opus) - Builds a 360° candidate profile through conversation
2. **serendipity-explorer** (Opus) - Interactive exploration of career directions
3. **role-discovery** (Opus) - Finds specific roles matching the profile
4. **job-search-agent** (Sonnet) - Direct job searching across boards

The core insight driving this project: **Most people undervalue their unique capabilities and can't articulate what makes them special.** The system should uncover the "hidden gems" that enable matching to opportunities the candidate would never find themselves.

**Current State:** The pipeline works end-to-end but produces profiles that feel more like "polished sales pitches" than deep psychological portraits. The questioning is competent but predictable, unlikely to surface truly latent characteristics.

---

## Technology Stack

- **LLM:** Claude via AWS Bedrock
  - Discovery agents: Claude Opus 4.5 (`claude-opus-4-5-20251101`)
  - Job search: Claude Sonnet
- **Backend:** FastAPI + Python 3.x
- **Frontend:** Jinja2 templates + htmx + Tailwind CSS
- **Data Storage:** JSON files in `webapp/sessions/`
- **External APIs:** AWS Bedrock Converse API, LinkedIn (regex scraping of public profiles)
- **Required Environment:** `AWS_BEARER_TOKEN_BEDROCK`, `AWS_REGION` (default: us-east-1)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Webapp                        │
│  (sessions, streaming chat, progress tracking)           │
└───────────────────────┬─────────────────────────────────┘
                        │
    ┌───────────────────┼───────────────────┐
    ▼                   ▼                   ▼
┌─────────┐      ┌─────────────┐      ┌──────────┐
│ Stage 1 │ ───► │   Stage 2   │ ───► │ Stage 3  │
│Pathfinder│      │  Explorer   │      │ Discovery│
└─────────┘      └─────────────┘      └──────────┘
    │                   │                   │
    ▼                   ▼                   ▼
candidate_         possibility_        role_discovery_
profile.md          map.md              *.md
```

**Data Flow:**
1. User provides LinkedIn URL or CV upload
2. Pathfinder conducts discovery conversation → outputs profile
3. Explorer takes profile → interactive exploration → outputs possibility map
4. Role-discovery takes profile + map → searches → outputs role recommendations

---

## Core Components: The Agents

### 1. Career-Pathfinder (The Profiler)

**Purpose:** Build a comprehensive 360° candidate profile through deep discovery.

**Current Methodology - "The 360° Framework":**
- Background: Career trajectory
- Skills Inventory: Technical and soft skills
- Energy Mapping: What energizes vs. drains
- Values & Priorities: What matters most
- Hidden Strengths: Capabilities they undervalue
- Work Style: Preferences and patterns
- Constraints: Geographic, compensation, lifestyle
- Aspirations: What they want

**Current Question Types (from the prompt):**

```markdown
### Career Journey Questions
- "Walk me through your career - not the resume version, but the real story"
- "What job or project changed the trajectory of your career?"
- "What's a role you took that looked like a step backward but wasn't?"

### Impact Questions
- "Tell me about a project where you felt you made a real difference"
- "When colleagues come to you for help, what do they usually need?"
- "What's something you've built or created that you're genuinely proud of?"

### Energy Questions
- "Describe your ideal Tuesday. What would you be doing hour by hour?"
- "What tasks do you procrastinate on, even when they're important?"
- "When do you enter 'flow state' at work?"

### Hidden Skill Questions
- "What do people compliment you on that seems obvious to you?"
- "What's something you do that your job description never mentioned?"
- "What skill do you have that most people in your field don't?"

### Values Questions
- "What's non-negotiable in your work life?"
- "What would make you turn down an otherwise perfect opportunity?"
- "What does success look like to you - not title or money, but how you feel?"
```

**Profile Output Structure (12 sections):**
1. Executive Summary
2. Career Narrative
3. Superpowers (3-5)
4. Skills Inventory
5. Energy Map
6. Hidden Value Propositions
7. Values & Non-Negotiables
8. Work Style Profile
9. Constraints
10. Aspirations
11. **Serendipity Signals** ← Key for downstream matching
12. Key Quotes

**The "Serendipity Signals" Section (from prompt):**
```markdown
### 11. Serendipity Signals
What makes this person unusual and where that's valuable:
- **Unexpected Skill Combinations**: Rare intersections that open non-obvious doors
- **Pattern Recognition**: "People with your profile often thrive in..."
- **Hidden Market Value**: Skills that are scarce in surprising places
- **Bridge Potential**: Where they could uniquely connect worlds
```

---

### 2. Serendipity-Explorer (The Possibility Mapper)

**Purpose:** Interactive exploration to discover non-obvious career directions.

**Methodology - Three Phases:**

**Phase 1: EXPAND (Diverge)**
- Surface 5-6 provocative directions they haven't considered
- Use "Adjacent Possibilities" - what's one skill/step away
- Use "Collision Detection" - where multiple trends intersect with their profile
- Frame as provocations: "What if you leaned hard into X?"

**Phase 2: EXPLORE (Go Deep)**
- Run what-if scenarios on hot directions
- Map adjacent moves: "This is 1 step away, but 2 steps gets you to Z"
- Test collision points: "Your A + B + C is rare. Three places that matters..."
- Track which branches are alive vs. dead

**Phase 3: CRYSTALLIZE (Converge)**
- Synthesize patterns across exploration
- Build the possibility map with paths, dead ends, unlock moves
- Name tradeoffs for each path
- Include a "wildcard" - the scary option that kept coming up

**Output - The Possibility Map:**
```markdown
# POSSIBILITY MAP

## What Makes You Unusual
[2-3 sentences on their distinctive collision point]

## PATH A: [Name]
| | |
|---|---|
| **What it looks like** | ... |
| **What unlocks it** | ... |
| **Energy level** | 🔥🔥🔥 High / 🔥🔥 Medium |
| **The tradeoff** | ... |
| **First move** | ... |

## DEAD ENDS
| Path | Why it's dead |
|------|---------------|
| X | [Reason] |

## UNLOCK MOVES
*Small actions that open disproportionate doors*

## PATTERNS I NOTICED
- [Observation about their energy]
- [What they consistently avoid]
```

---

### 3. Role-Discovery (The Matcher)

**Purpose:** Find specific roles and companies that match the profile.

**Inputs:**
- Required: Candidate profile (from pathfinder)
- Optional: Possibility map (from explorer) - if present, focuses on high-energy paths

**Output:** Role Discovery Report with:
- 3-5 target role recommendations with specific companies
- Anti-targets (roles to avoid based on energy map/values)
- Positioning themes
- Immediate actions

---

## Real Output Example: Profile Quality Assessment

Here's the actual profile generated for a test user (enterprise sales leader at AWS):

**What the current system captured well:**
- Career narrative and progression
- Explicit skills and domain expertise
- Energy map (what energizes vs. drains)
- Compensation requirements and constraints
- A unique trait: "vibe coding" as hands-on AI experimentation

**The "Serendipity Signals" section as generated:**
```markdown
### 4.1 AI Power User Intuition
His side projects show what's possible when a curious non-engineer uses
AI coding tools. He built working systems not by designing architecture,
but by experimenting. This gives him:
- Real experience as an AI tool user (not just seller)
- Intuition for what AI can actually do vs. marketing hype
- Ability to demo and discuss AI workflows authentically
```

**What's MISSING from the profile:**
- **Cognitive style**: How do they actually think? Analytical vs. intuitive? Big picture vs. detail?
- **Decision-making patterns**: Risk tolerance, information needs before acting
- **Failure modes**: What happens when they're stressed, overwhelmed, bored?
- **What they avoid**: The negative space that reveals preferences
- **Relationship to authority/structure**: Beyond "dislikes bureaucracy"
- **Identity vs. capability**: What they see as core to who they ARE vs. just skills
- **Unconscious patterns**: Themes they don't notice in their own stories
- **What frustrates them about others**: Reveals their own values
- **Learning style**: How they acquire new capabilities
- **Collaboration archetypes**: Leader, supporter, lone wolf, orchestrator?

---

## Key Code Patterns

### Progress Tracking System

Agents report progress via HTML comments:
```python
# Agent includes at end of each response:
<!--progress:45|topics:background,skills,energy-->

# Backend parsing in app.py:
def parse_progress_tag(text: str) -> tuple[int, list[str]]:
    match = re.search(r'<!--progress:(\d+)\|topics:([^>]+)-->', text)
    if match:
        return int(match.group(1)), match.group(2).split(',')
    return 0, []
```

### Session Context Management

The webapp maintains conversation context per stage:
```python
# Each stage has its own chat history
session = {
    "stage": "pathfinder",
    "pathfinder_messages": [...],  # Full conversation
    "explorer_messages": [...],
    "discovery_messages": [...],
    "outputs": {
        "profile": "...",        # Generated at stage completion
        "possibility_map": "...",
        "role_discovery": "..."
    }
}
```

### Context Summarization System

Long conversations are automatically summarized to stay within token limits:
```python
MAX_RECENT_MESSAGES = 15  # Keep full context for last N messages
SUMMARY_THRESHOLD = 20    # Start summarizing when conversation exceeds this

def summarize_old_messages(messages, keep_recent=15):
    """Summarize older messages to reduce token usage."""
    old_messages = messages[:-keep_recent]
    recent_messages = messages[-keep_recent:]
    # Build condensed summary of old messages
    # Prepend summary to system prompt
```
This prevents token limit issues in longer discovery sessions while maintaining recent context.

### Agent Prompt Loading

Agents are defined as YAML frontmatter + markdown:
```yaml
---
name: career-pathfinder
description: "Use this agent to build a 360° candidate profile..."
model: opus
color: cyan
---

[Agent prompt body in markdown]
```

---

## Current Implementation Details: Questioning Flow

**How questions are structured (from career-pathfinder prompt):**

```markdown
## Your Questioning Approach

You ask questions in strategic clusters, not rapid-fire lists.
Each question builds on previous answers. Dig deep - surface-level
answers aren't enough.

## Your Conversation Style

- Be warm but direct
- Use their language back to them
- Challenge assumptions gently: "You said you're 'just' a developer.
  But you also mentioned leading that architecture overhaul..."
- Dig deeper: "Tell me more about that." "What specifically did you do?"
- Reflect patterns: "I'm noticing a theme here..."
- **ALWAYS number your questions** when asking multiple questions in one response
```

**Note:** All agents now enforce numbered questions when asking multiple questions. This was added to improve conversation flow.

**The problem:** Despite these instructions, the actual questions are:
1. **Direct and predictable** - ask about X, get answer about X
2. **Consciously accessible** - user can easily answer because they already know
3. **Forward-looking** - "What do you want?" vs. revealing hidden patterns
4. **Skills-focused** - what you CAN do vs. what you're DRAWN to do

---

## Known Issues & Pain Points

### Issue 1: Questions Don't Surface Latent Traits

Current questions ask what users consciously know:
- "What energizes you?" → User reports what they think energizes them
- "What are your superpowers?" → User reports what they believe are superpowers

**What's missing:** Indirect elicitation techniques that reveal patterns the user doesn't see themselves.

### Issue 2: Profile Structure Drives Predictable Output

The 12-section profile template creates pressure to fill slots:
- "I need to find 3-5 superpowers"
- "I need an energy map with + and - items"

This leads to **slot-filling** rather than **emergent understanding**.

### Issue 3: "Serendipity" Is Bolted On, Not Emergent

The "Serendipity Signals" section exists but:
- Questions don't specifically surface unusual combinations
- The agent is instructed to find them, but with what data?
- Results read as clever observations rather than discovered surprises

### Issue 4: No Behavioral Data, Only Self-Report

Everything in the profile comes from what the user says about themselves:
- No analysis of HOW they tell stories (communication style)
- No inference from WHAT they choose to share (priorities revealed by omission)
- No behavioral patterns from described situations (how they actually act)

### Issue 5: Downstream Agents Inherit Limitations

If the profile is shallow, serendipity-explorer and role-discovery can only work with shallow data. They can't uncover what wasn't discovered upstream.

---

## Constraints & Requirements

**Technical Constraints:**
- Must work via text conversation (no video/audio analysis)
- Must produce results in a single session (can't observe over time)
- Must work with Claude API (no fine-tuning, no custom models)
- Must keep conversations reasonable length (cost and user patience)

**User Experience Constraints:**
- Questions should feel natural, not like a psychological assessment
- Users should feel helped, not analyzed
- Process should respect user's time (can't do 4-hour discovery sessions)

**Output Constraints:**
- Profile must be usable by downstream agents (structured, clear)
- Profile must be readable by humans (the user wants to see it)
- Profile must capture enough for serendipitous matching

---

## Specific Questions for Consultants

### On Questioning Methodology

1. **What questioning techniques from psychology/coaching could surface traits users don't consciously know?** Current questions are direct (ask X, get X). What indirect techniques reveal patterns users don't see in themselves? Consider: projective techniques, behavioral interviewing, values elicitation through forced choices, story analysis.

2. **How can we infer psychological traits from HOW someone answers, not just WHAT they answer?** Their word choices, what they emphasize, what they skip, emotional tone, level of detail - these reveal more than the content itself.

3. **What "negative space" questions reveal preferences through avoidance?** Instead of "what do you want?", what questions reveal preferences through what someone consistently avoids, rejects, or doesn't mention?

### On Profile Structure

4. **Should the profile structure be emergent rather than templated?** Current approach: fill these 12 sections. Alternative: let the profile structure emerge from what's actually discovered. What are the tradeoffs?

5. **What profile elements are most predictive of non-obvious job fit?** If we could only capture 5 things, which 5 would enable the most serendipitous matching? Current profile has 12 sections - which actually matter for discovering unexpected fits?

6. **How do we capture "what makes someone unusual" without forcing unusual interpretations?** The current system has a "Serendipity Signals" section, but it often reads as reaching for uniqueness. How do we surface genuine distinctiveness?

### On Serendipitous Discovery

7. **What creates genuine serendipity in career matching?** The goal is finding jobs users don't know exist but fit perfectly. What profile data enables this? What matching logic? What examples exist of systems that achieved genuine serendipity?

8. **How do we distinguish "surprising" from "irrelevant"?** A surprising match should be unexpected but obviously right in hindsight. What keeps recommendations from being merely random or stretching too far?

9. **Should the system look for "latent roles" - jobs that don't exist yet but match the profile?** E.g., "Based on your profile, companies will need someone who can X + Y + Z in 2 years - you could define that role."

### On Multi-Agent Pipeline

10. **Is the profile → exploration → matching pipeline optimal?** Current flow separates "understanding who you are" from "exploring directions" from "finding specific roles." Should these be more integrated? Less separated?

---

## Appendix: Full Agent Prompt (Career-Pathfinder)

```markdown
You are an elite Career Discovery Specialist focused exclusively on
understanding candidates at a deep level. Your job is to conduct a
thorough 360° assessment of who someone is professionally - their
skills, strengths, values, energy patterns, hidden capabilities,
and what makes them unique.

**IMPORTANT: You do NOT recommend roles, companies, or action plans.
Your sole output is a comprehensive Candidate Profile.**

## Your Core Philosophy

Before someone can find the right opportunity, they need to truly
understand themselves. Most people undervalue their unique capabilities
and can't articulate what makes them special. Your job is to uncover
the full picture - the obvious skills AND the hidden gems.

## Your Methodology: The 360° Framework

**Background**: Full career trajectory - not just titles, but what they did
**Skills Inventory**: Technical and soft skills, including ones they take for granted
**Energy Mapping**: What work energizes vs. drains them - critical for fit
**Values & Priorities**: What matters most in work and life
**Hidden Strengths**: Capabilities others see in them that they undervalue
**Work Style**: How they prefer to work, collaborate, communicate
**Constraints**: Geographic, compensation, lifestyle factors
**Aspirations**: What they want their professional life to look like

## Your Questioning Approach

You ask questions in strategic clusters, not rapid-fire lists. Each
question builds on previous answers. Dig deep - surface-level answers
aren't enough.

[Full question list included in main document above]

## Critical Rules

- Always explain what you're learning and why it matters
- Push past surface answers - the gold is in the specifics
- Capture their voice - use their words in the profile
- Be thorough - a thin profile is useless
```

---

## Appendix: Example Profile Sections

**Superpowers section (current output):**
```markdown
### 2.1 The Curious Experimenter
Unlike typical enterprise salespeople who stay in their lane, Guy gets
his hands dirty with technology. He uses AI coding tools (Claude Code)
to build real projects - not because he can architect systems himself,
but because he's genuinely curious and wants to understand what's possible.

**Why this matters:** He can have authentic conversations with technical
buyers because he's actually used the tools.
```

**Energy Map (current output):**
```markdown
### ENERGIZERS (Seek roles that maximize these)
| Activity | Energy Level |
|----------|--------------|
| First and second customer meetings | +++ |
| Analyzing what customers actually need | +++ |
| Being "the closer" - the person customers want | +++ |

### DRAINS (Avoid roles heavy in these)
| Activity | Energy Level |
|----------|--------------|
| Bureaucracy and red tape | --- |
| Product roadmap/feedback loops | -- |
```

---

*Generated by project-briefer agent for expert LLM consultation.*
