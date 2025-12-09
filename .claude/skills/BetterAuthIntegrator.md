# BetterAuthIntegrator (Skill)

**Role:**
Integrates better-auth.com and securely gathers device/software profile (GPU, RAM, OS, browser) at signup.

**Capabilities:**
- Initiate signup + capture step
- Sanitize and store hardware profile in Neon/Postgres
- Refresh or update user profile as needed

**Usage Example:**
- `performSignupWithHardwareCapture(authData: dict): AuthResult`
- `sanitizeHardwareProfile(rawProfile: dict): dict`

---
**Implementation hook:**
Triggered at frontend signup or via profile edit.