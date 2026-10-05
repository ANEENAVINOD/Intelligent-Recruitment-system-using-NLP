import pandas as pd
import streamlit as st

from utils import (
    EDU_LEVELS,
    LEVEL_NAMES,
    clean_text,
    extract_education_level,
    extract_email,
    extract_experience_years,
    extract_phone,
    extract_text_from_pdf,
    find_skills,
    parse_skills,
    responsibility_scores,
)

st.set_page_config(page_title="Resume Analyzer", layout="wide")
st.title("Resume Analyzer")

# ---------------- Sidebar: recruiter inputs ----------------
st.sidebar.header("Job requirements")

skills_input = st.sidebar.text_area(
    "Required skills (comma-separated)",
    placeholder="python, sql, machine learning, docker",
)
min_exp = st.sidebar.number_input("Minimum experience (years)", 0.0, 40.0, 0.0, 0.5)
edu_choice = st.sidebar.selectbox("Minimum education", list(EDU_LEVELS.keys()))
job_text = st.sidebar.text_area(
    "Job responsibilities / description",
    height=200,
    placeholder="Paste the responsibilities or the full job description here...",
)

st.sidebar.subheader("Importance (weights)")
w_skills = st.sidebar.slider("Skills", 0, 100, 40)
w_exp = st.sidebar.slider("Experience", 0, 100, 20)
w_edu = st.sidebar.slider("Education", 0, 100, 15)
w_resp = st.sidebar.slider("Responsibilities match", 0, 100, 25)

# ---------------- Main area ----------------
uploaded_files = st.file_uploader(
    "Upload resumes (PDF)", type=["pdf"], accept_multiple_files=True
)
run = st.button("Rank candidates", type="primary")

if run:
    if not uploaded_files:
        st.warning("Please upload at least one resume.")
        st.stop()

    required_skills = parse_skills(skills_input)
    required_edu = EDU_LEVELS[edu_choice]

    # Only criteria the recruiter actually filled in count towards the score
    weights = {
        "skills": w_skills if required_skills else 0,
        "exp": w_exp if min_exp > 0 else 0,
        "edu": w_edu if required_edu > 0 else 0,
        "resp": w_resp if job_text.strip() else 0,
    }
    total_weight = sum(weights.values())
    if total_weight == 0:
        st.warning("Fill in at least one requirement (with a weight above 0) in the sidebar.")
        st.stop()

    # Read all resumes
    names, texts = [], []
    for f in uploaded_files:
        names.append(f.name)
        texts.append(clean_text(extract_text_from_pdf(f)))

    resp_scores = responsibility_scores(job_text, texts)

    results = []
    for name, text, resp in zip(names, texts, resp_scores):
        matched, missing = find_skills(text, required_skills)
        skill_score = len(matched) / len(required_skills) if required_skills else 0

        years = extract_experience_years(text)
        exp_score = min(years / min_exp, 1.0) if min_exp > 0 else 0

        edu_level = extract_education_level(text)
        edu_score = min(edu_level / required_edu, 1.0) if required_edu > 0 else 0

        final = (
            weights["skills"] * skill_score
            + weights["exp"] * exp_score
            + weights["edu"] * edu_score
            + weights["resp"] * resp
        ) / total_weight * 100

        results.append({
            "File": name,
            "Score": round(final, 1),
            "Email": extract_email(text),
            "Phone": extract_phone(text),
            "Skills matched": f"{len(matched)}/{len(required_skills)}" if required_skills else "-",
            "Experience (yrs)": years,
            "Education": LEVEL_NAMES[edu_level],
            "Responsibility match %": round(resp * 100),
            "Matched skills": ", ".join(matched),
            "Missing skills": ", ".join(missing),
        })

    df = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)

    st.subheader("Ranked candidates")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100, format="%.1f"
            ),
        },
    )

    st.subheader("Why each candidate got their score")
    for _, row in df.iterrows():
        with st.expander(f"#{row['Rank']}  {row['File']}  -  {row['Score']}"):
            st.write(f"**Matched skills:** {row['Matched skills'] or 'None'}")
            st.write(f"**Missing skills:** {row['Missing skills'] or 'None'}")
            st.write(f"**Experience found:** {row['Experience (yrs)']} years")
            st.write(f"**Highest education found:** {row['Education']}")
            st.write(f"**Responsibility match:** {row['Responsibility match %']}%")

    st.download_button(
        "Download results (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        "ranked_candidates.csv",
        "text/csv",
    )
