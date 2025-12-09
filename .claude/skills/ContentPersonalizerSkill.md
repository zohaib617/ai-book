# ContentPersonalizerSkill (Skill)

**Role:**
Dynamically personalizes textbook chapters/sections to user’s hardware profile, especially for install/setup/workload parts.

**Capabilities:**
- Rewrite or annotate content based on hardware constraints
- Insert clarifications for simulation, rendering, or system requirements

**Usage Example:**
- `personalizeContent(mdContent: string, hardwareProfile: dict): PersonalizedContent`
- `highlightUnsupportedSections(mdContent: string, hardwareProfile: dict): MDXDiff`

---
**Implementation hook:**
Invoked per-user-view in frontend or on backend render.