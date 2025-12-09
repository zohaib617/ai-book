# UrduTranslationSkill (Skill)

**Role:**
Translates educational MDX to Urdu, preserving technical terms/code, and handles RTL/Unicode.

**Capabilities:**
- Prose-only translation; leaves code and terms in English
- Output correct Urdu script, alt text, warnings on ambiguity

**Usage Example:**
- `translateSection(mdContent: string): UrduMDXContent`
- `preserveCodeAndTerms(orig: string, lang: str="ur"): string`

---
**Implementation hook:**
Invoked by user toggle or backend frontend API route.