import os
import json
import re
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load variables from .env into the environment (expects OPENAI_API_KEY=... in .env)
load_dotenv()

# Hard cap on profile length (characters, including spaces & punctuation)
MAX_PROFILE_CHARS = 500

# 1. Page Configuration & Title
st.set_page_config(page_title="Business Profile Generator", layout="centered")
st.title("Business Profile Generator")
st.write("Answer a few quick questions to generate a professional company profile.")

# 2. Load Industry Taxonomy (CSV -> dict of {Primary Industry: [Sub-Industries]})
@st.cache_data
def load_industry_taxonomy(csv_path: str = "sa_industry_taxonomy.csv"):
    df = pd.read_csv(csv_path)
    taxonomy = (
        df.groupby("Primary Industry")["Sub-Industry"]
        .apply(list)
        .to_dict()
    )
    return taxonomy

# 2b. Load Terms & Conditions (plain text, editable without code changes)
@st.cache_data
def load_terms_and_conditions(path: str = "terms_and_conditions.txt"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return (
            "1. Profile must not exceed 500 characters.\n"
            "2. No false, misleading, or unverifiable claims.\n"
            "3. No unsupported guarantees or superlatives.\n"
            "4. No discriminatory, offensive, or unlawful content.\n"
            "5. No medical, legal, or financial advice or outcome promises.\n"
            "6. Professional tone; no profanity or placeholders."
        )

industry_taxonomy = load_industry_taxonomy()
primary_industries = sorted(industry_taxonomy.keys())
terms_and_conditions = load_terms_and_conditions()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def enforce_char_limit(text: str, limit: int = MAX_PROFILE_CHARS) -> str:
    """Trim text to <= limit chars, cutting cleanly at a sentence or word
    boundary rather than mid-word. Safety net for when the LLM overshoots."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text

    window = text[:limit]

    # Prefer to end on the last sentence terminator within the window.
    last_sentence = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if last_sentence >= int(limit * 0.6):
        return window[: last_sentence + 1].strip()

    # Otherwise cut at the last whole word.
    last_space = window.rfind(" ")
    if last_space > 0:
        return window[:last_space].strip()

    # No boundary found — hard cut.
    return window.strip()


def char_count_caption(text: str, limit: int = MAX_PROFILE_CHARS):
    """Render a live character counter that turns red when over the limit."""
    count = len(text or "")
    if count > limit:
        st.markdown(
            f"<span style='color:#c0392b;font-weight:600;'>"
            f"{count} / {limit} characters — over the limit by {count - limit}."
            f"</span>",
            unsafe_allow_html=True,
        )
    else:
        st.caption(f"{count} / {limit} characters")


def run_compliance_check(profile_text: str, terms: str, api_key: str) -> dict:
    """Agent 2 — Compliance Reviewer. Returns a structured dict:
    {
      "verdict": "compliant" | "needs_review",
      "issues": [
        {"rule": str, "sentence": str, "reason": str,
         "suggestion": str, "severity": "low"|"medium"|"high"}
      ]
    }
    The agent must only quote sentences that literally appear in the profile.
    """
    # Deterministic length check (never trust an LLM to count characters).
    length_issue = None
    if len(profile_text) > MAX_PROFILE_CHARS:
        length_issue = {
            "rule": "1. Length — max 500 characters (including spaces)",
            "sentence": profile_text[MAX_PROFILE_CHARS:].strip()[:120],
            "reason": (
                f"The profile is {len(profile_text)} characters, which exceeds "
                f"the {MAX_PROFILE_CHARS}-character limit by "
                f"{len(profile_text) - MAX_PROFILE_CHARS}."
            ),
            "suggestion": "Shorten the profile to 500 characters or fewer.",
            "severity": "high",
        }

    reviewer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)

    review_template = """
    You are a compliance reviewer for public business profiles. Check the
    PROFILE below against the TERMS & CONDITIONS. You flag potential issues
    only; a human team makes the final decision, so be precise and avoid
    inventing problems.

    TERMS & CONDITIONS:
    {terms}

    PROFILE TO REVIEW:
    {profile}

    Return ONLY valid JSON (no markdown, no commentary) in exactly this shape:
    {{
      "verdict": "compliant" | "needs_review",
      "issues": [
        {{
          "rule": "the specific rule number and short title it relates to",
          "sentence": "the exact sentence or phrase from the PROFILE that triggered this — copied verbatim, never paraphrased or invented",
          "reason": "a short explanation of why it is a potential issue",
          "suggestion": "a short suggested rewrite or fix",
          "severity": "low" | "medium" | "high"
        }}
      ]
    }}

    Rules for your output:
    - The "sentence" field MUST be text that appears verbatim in the PROFILE.
      If you cannot quote it exactly, do not raise that issue.
    - Do NOT check the 500-character length rule; that is handled separately.
    - If there are no issues, return "verdict": "compliant" and an empty
      "issues" list.
    """

    prompt = PromptTemplate.from_template(review_template)
    chain = prompt | reviewer_llm | StrOutputParser()

    result = {"verdict": "compliant", "issues": []}
    try:
        raw = chain.invoke({"terms": terms, "profile": profile_text})
        # Strip code fences if the model added them despite instructions.
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            result["verdict"] = parsed.get("verdict", "needs_review")
            issues = parsed.get("issues", [])
            if isinstance(issues, list):
                result["issues"] = issues
    except (json.JSONDecodeError, Exception) as e:  # noqa: BLE001
        result["verdict"] = "needs_review"
        result["issues"] = [{
            "rule": "Reviewer error",
            "sentence": "",
            "reason": f"The compliance check could not be parsed automatically: {e}. "
                      f"Please review this profile manually.",
            "suggestion": "Re-run the check or review by hand.",
            "severity": "medium",
        }]

    # Prepend the deterministic length issue if present.
    if length_issue:
        result["issues"].insert(0, length_issue)
        result["verdict"] = "needs_review"

    return result


def render_compliance_panel(review: dict):
    """Render the green 'compliant' box or the red flagged-issues panel."""
    issues = review.get("issues", [])
    if review.get("verdict") == "compliant" and not issues:
        st.success("✓ Looks compliant — no issues flagged against the terms & conditions.")
        return

    st.error(f"⚠️ Potential T&C issues flagged for review ({len(issues)}). "
             f"A human team should confirm each item.")

    severity_color = {"high": "#c0392b", "medium": "#e67e22", "low": "#f1c40f"}
    for i, issue in enumerate(issues, start=1):
        sev = str(issue.get("severity", "medium")).lower()
        color = severity_color.get(sev, "#e67e22")
        with st.container(border=True):
            st.markdown(
                f"**{i}. {issue.get('rule', 'Unspecified rule')}** "
                f"<span style='color:{color};font-weight:600;'>[{sev.upper()}]</span>",
                unsafe_allow_html=True,
            )
            sentence = issue.get("sentence", "")
            if sentence:
                st.markdown("**Flagged text:**")
                st.markdown(f"> {sentence}")
            if issue.get("reason"):
                st.markdown(f"**Why:** {issue['reason']}")
            if issue.get("suggestion"):
                st.markdown(f"**Suggested fix:** {issue['suggestion']}")


# 3. Initialize Session States (Needed to preserve data across user edits and clicks)
if "generated_profile" not in st.session_state:
    st.session_state.generated_profile = ""
if "is_editing" not in st.session_state:
    st.session_state.is_editing = False
if "compliance_review" not in st.session_state:
    st.session_state.compliance_review = None

# 4. API Configuration (loaded from .env, not entered by the user)
openai_api_key = os.getenv("OPENAI_API_KEY")

with st.sidebar:
    st.header("API Setup")
    if openai_api_key:
        st.success("OpenAI API key loaded from environment.")
    else:
        st.error("No OPENAI_API_KEY found. Add it to your .env file, e.g.:\nOPENAI_API_KEY=sk-...")

    st.divider()
    st.header("Terms & Conditions")
    st.caption("Rules the profile is generated against and checked for. "
               "Edit terms_and_conditions.txt to change them.")
    with st.expander("View current T&Cs"):
        st.text(terms_and_conditions)

# 5. Questionnaire — Industry dropdowns live OUTSIDE the form so selecting a
#    Primary Industry immediately reruns the app and filters the Sub-Industry
#    options (widgets inside st.form only update on submit, so cascading
#    dropdowns can't live inside the form).
st.subheader("Step 1: Tell us about your business")

biz_industry = st.selectbox(
    "2. Primary Industry:",
    options=primary_industries,
    index=None,
    placeholder="Select a primary industry...",
)

sub_industry_options = industry_taxonomy.get(biz_industry, []) if biz_industry else []
biz_sub_industry = st.selectbox(
    "3. Sub-Industry / Niche:",
    options=sub_industry_options,
    index=None,
    placeholder="Select a primary industry first..." if not biz_industry else "Select a sub-industry...",
    disabled=not biz_industry,
)

with st.form("business_questions"):
    biz_name = st.text_input("1. Business Name:", placeholder="e.g., Apex Plumbing Solutions")
    biz_audience = st.text_input("4. Target Audience:", placeholder="e.g., Homeowners and local property managers")
    biz_usps = st.text_area("5. Key Strengths / Unique Value Proposition:", placeholder="e.g., 24/7 emergency response, eco-friendly fixtures, 10-year warranty")

    col1, col2 = st.columns(2)
    with col1:
        biz_area = st.text_input("6. Location / Service Area:", placeholder="e.g., Sandton, Johannesburg")
    with col2:
        biz_reach = st.selectbox(
            "Reach:",
            options=["Local", "Provincial", "National", "Online-only"],
            index=None,
            placeholder="Select reach...",
        )

    submit_button = st.form_submit_button("Generate Profile Summary")

# 6. LLM Chain Logic & Generation Trigger
if submit_button:
    if not openai_api_key:
        st.error("Please provide a valid OpenAI API key in the sidebar.")
    elif not biz_name or not biz_industry:
        st.warning("⚠️ Please provide at least a Business Name and Industry to proceed.")
    else:
        with st.spinner("AI is drafting your professional profile..."):
            try:
                # Initialize the LLM (Using gpt-4o-mini for fast, cost-effective generation)
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=openai_api_key)

                # Combine location fields into one readable string for the prompt
                if biz_area and biz_reach:
                    location_str = f"{biz_area} ({biz_reach} reach)"
                elif biz_area:
                    location_str = biz_area
                elif biz_reach:
                    location_str = f"{biz_reach} reach"
                else:
                    location_str = "Not specified"

                # Define a structured prompt layout. The T&Cs are injected so the
                # copywriter agent stays within the rules up front (prevention).
                template = """
                You are an expert business copywriter. Create a professional, compelling, and engaging 
                business profile summary based exactly on the questionnaire data provided below.

                You MUST comply with the following TERMS & CONDITIONS. Do not produce any
                content that would violate them:
                {terms}

                Business Questionnaire Answers:
                1. Company Name: {name}
                2. Industry: {industry}
                3. Sub-Industry: {sub_industry}
                4. Target Audience: {audience}
                5. Key Strengths/USPs: {usps}
                6. Location/Service Area: {location}
                
                Instructions for Output:
                - HARD LIMIT: the summary MUST be 500 characters or fewer, including spaces.
                  This is a strict technical constraint — stay well within it.
                - Write a single concise, professional summary (roughly 2-4 sentences).
                - Tone: Trustworthy, modern, and customer-centric.
                - Focus on how the business solves problems for its target audience using its unique strengths.
                - Do not use placeholders or robotic phrasing.
                - Do not make unverifiable claims, unsupported guarantees, or superlatives.
                """

                prompt = PromptTemplate.from_template(template)

                # Chain setup using LangChain Expression Language (LCEL)
                chain = prompt | llm | StrOutputParser()

                # Invoke the chain with form values
                result = chain.invoke({
                    "terms": terms_and_conditions,
                    "name": biz_name,
                    "industry": biz_industry,
                    "sub_industry": biz_sub_industry,
                    "audience": biz_audience,
                    "usps": biz_usps,
                    "location": location_str,
                })

                # Code-level guard: enforce the 500-char cap even if the LLM overshoots.
                result = enforce_char_limit(result)

                # Store the result in session state and reset edit flags
                st.session_state.generated_profile = result
                st.session_state.is_editing = False

                # Run the compliance agent on the freshly generated profile.
                with st.spinner("Compliance agent is reviewing the draft..."):
                    st.session_state.compliance_review = run_compliance_check(
                        result, terms_and_conditions, openai_api_key
                    )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# 7. Step 2: Displaying and Allowing Edits on the Generated Profile
if st.session_state.generated_profile:
    st.divider()
    st.subheader("Step 2: Review and Customise Your Profile")

    # Toggle editing UI based on button clicks
    if st.session_state.is_editing:
        # If editing, show a text area initialized with the current session state text
        edited_text = st.text_area(
            "Modify your profile summary below (max 500 characters):",
            value=st.session_state.generated_profile,
            height=200,
            max_chars=MAX_PROFILE_CHARS,
        )
        char_count_caption(edited_text)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Changes", type="primary"):
                # Guard length on save, then re-run compliance on the edited text.
                saved = enforce_char_limit(edited_text)
                st.session_state.generated_profile = saved
                st.session_state.is_editing = False
                if openai_api_key:
                    with st.spinner("Compliance agent is re-checking your edits..."):
                        st.session_state.compliance_review = run_compliance_check(
                            saved, terms_and_conditions, openai_api_key
                        )
                st.rerun()
        with col2:
            if st.button("Cancel"):
                st.session_state.is_editing = False
                st.rerun()
    else:
        # If not editing, display the text nicely formatted as markdown
        st.markdown(st.session_state.generated_profile)
        char_count_caption(st.session_state.generated_profile)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Edit Profile Summary"):
                st.session_state.is_editing = True
                st.rerun()
        with col2:
            if st.button("Check Against Terms & Conditions"):
                if openai_api_key:
                    with st.spinner("Compliance agent is reviewing..."):
                        st.session_state.compliance_review = run_compliance_check(
                            st.session_state.generated_profile,
                            terms_and_conditions,
                            openai_api_key,
                        )
                    st.rerun()
                else:
                    st.error("Please provide a valid OpenAI API key in the sidebar.")
        with col3:
            # Provide a direct download button for convenience
            st.download_button(
                label="Download Profile (.txt)",
                data=st.session_state.generated_profile,
                file_name=f"{biz_name.lower().replace(' ', '_')}_profile.txt" if biz_name else "profile.txt",
                mime="text/plain"
            )

    # 8. Compliance review panel (flagged items box)
    if st.session_state.compliance_review is not None:
        st.divider()
        st.subheader("Compliance Review")
        st.caption("Flagged by the compliance agent for human review. "
                   "This is guidance, not a legal decision.")
        render_compliance_panel(st.session_state.compliance_review)
