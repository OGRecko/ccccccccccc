# Browser profiles

One folder per account, each a full persistent Chromium profile (cookies, logins,
localStorage). GARVIS never reads these files - the browser does, and sessions
persist so you only log in once per profile, by hand, including 2FA.

Naming: create a folder whose name is the label you will use in speech, e.g.

    profiles/
      default/
      work/
      personal/
      shop-account-b/

Commands (stage 6):

    "Garvis, open github in my work profile"

You can also tell GARVIS which profile to use by default in config.yaml
(`browser.default_profile`).

Do NOT commit these folders - they contain live session cookies.
They are gitignored on purpose (see .gitignore).
