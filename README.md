# Business Profile Generator

A Streamlit app that generates a concise (<= 500 character) business profile
summary from a short questionnaire, then runs a second "compliance" agent that
reviews the text against a set of terms & conditions and flags potential issues
for human review.

## How it works

1. **Copywriter agent** generates the profile. The terms & conditions are
   injected into its prompt so it stays within the rules, and the output is
   capped at 500 characters.
2. **Compliance agent** reviews the generated or user-edited text against
   `terms_and_conditions.txt` and returns structured findings: the rule, the
   exact offending sentence, a reason, a suggested fix, and a severity.
3. Flagged items appear in a review panel. This is guidance, not a legal
   decision — a human team confirms each item.

## Running locally

```
pip install -r requirements.txt
# create a .env file containing: OPENAI_API_KEY=sk-...
streamlit run app.py
```

## Deploying on Streamlit Community Cloud

1. Push this repo to GitHub.
2. At https://share.streamlit.io, create a new app pointing at this repo and
   `app.py`.
3. In the app's **Settings -> Secrets**, add:
   ```
   OPENAI_API_KEY = "sk-..."
   ```
4. Deploy. Your shareable link will be `https://<app-name>.streamlit.app`.

## Editing the rules

Edit `terms_and_conditions.txt` to change the rules the profile is generated
against and checked for. The app reads it at runtime.
