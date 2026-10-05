import re
from datetime import date

import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Education ----------
EDU_LEVELS = {"Any": 0, "Diploma": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
LEVEL_NAMES = {0: "Not found", 1: "Diploma", 2: "Bachelor's", 3: "Master's", 4: "PhD"}

EDU_PATTERNS = {
    4: r"\bph\.?\s?d\b|\bdoctorate\b",
    3: r"\bmaster'?s?\b|\bm\.?\s?tech\b|\bm\.?\s?sc\b|\bmba\b|\bmca\b",
    2: r"\bbachelor'?s?\b|\bb\.?\s?tech\b|\bb\.?\s?sc\b|\bb\.e\b|\bbca\b|\bb\.?\s?com\b|\bbba\b",
    1: r"\bdiploma\b",
}

# Lines containing these words are ignored when calculating work experience,
# so a "2018 - 2022" college date range is not counted as a job.
EDU_LINE_WORDS = re.compile(
    r"universit|college|school|b\.?\s?tech|bachelor|master|degree|diploma|"
    r"b\.?\s?sc|m\.?\s?sc|mba|cgpa|gpa|percentage",
    re.IGNORECASE,
)


# ---------- PDF and cleaning ----------
def extract_text_from_pdf(file):
    pdf_bytes = file.getvalue()  # getvalue() is safe across Streamlit reruns
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def clean_text(text):
    text = re.sub(r"[•●▪◦■►]", "-", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------- Basic details ----------
def extract_email(text):
    match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    return match.group(0).strip(".") if match else "Not found"


def extract_phone(text):
    candidates = re.findall(
        r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,5}\)?[\s.-]?)?\d{3,5}[\s.-]?\d{4,5}",
        text,
    )
    for c in candidates:
        digits = re.sub(r"\D", "", c)
        if 10 <= len(digits) <= 13:
            return c.strip()
    return "Not found"


# ---------- Skills (now driven by recruiter input) ----------
def parse_skills(raw):
    """'Python, SQL\nDocker' -> ['python', 'sql', 'docker']"""
    parts = re.split(r"[,\n;]", raw)
    skills = []
    for p in parts:
        p = p.strip().lower()
        if p and p not in skills:
            skills.append(p)
    return skills


def find_skills(text, required_skills):
    text_lower = text.lower()
    matched, missing = [], []
    for skill in required_skills:
        pattern = r"(?<![a-z0-9+#])" + re.escape(skill) + r"(?![a-z0-9+#])"
        if re.search(pattern, text_lower):
            matched.append(skill)
        else:
            missing.append(skill)
    return matched, missing


# ---------- Experience ----------
def extract_experience_years(text):
    """Returns the larger of: years stated in the resume, or years computed from date ranges."""
    text_l = text.lower()

    # 1) Explicit statements like "5 years of experience" / "experience of 3 yrs"
    stated = [
        float(x)
        for x in re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)[^.\n]{0,30}experience", text_l)
    ]
    stated += [
        float(x)
        for x in re.findall(r"experience[^.\n]{0,20}?(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)", text_l)
    ]
    stated_years = max(stated) if stated else 0.0

    # 2) Date ranges like "2019 - 2023" or "Jan 2019 - Present"
    current_year = date.today().year
    range_pattern = re.compile(
        r"\b((?:19|20)\d{2})\s*(?:-|–|—|to)\s*(?:[A-Za-z]{3,9}\.?\s*)?"
        r"((?:19|20)\d{2}|present|current|now)\b",
        re.IGNORECASE,
    )
    intervals = []
    for line in text.split("\n"):
        if EDU_LINE_WORDS.search(line):
            continue
        for start, end in range_pattern.findall(line):
            s = int(start)
            e = current_year if end.lower() in ("present", "current", "now") else int(end)
            if e >= s:
                intervals.append((s, e))

    # Merge overlapping ranges so parallel jobs are not double counted
    intervals.sort()
    merged = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    computed_years = float(sum(e - s for s, e in merged))

    return round(max(stated_years, computed_years), 1)


# ---------- Education ----------
def extract_education_level(text):
    text_l = text.lower()
    for level in (4, 3, 2, 1):
        if re.search(EDU_PATTERNS[level], text_l):
            return level
    return 0


# ---------- Job responsibilities (text similarity) ----------
def responsibility_scores(job_text, resume_texts):
    """TF-IDF cosine similarity between the job description and each resume (0 to 1)."""
    if not job_text.strip() or not resume_texts:
        return [0.0] * len(resume_texts)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([job_text] + resume_texts)
    sims = cosine_similarity(matrix[0:1], matrix[1:])[0]
    # Raw cosine scores for resume-vs-JD are usually low (0.1 to 0.5),
    # so we scale them up. A score of 0.5 or more counts as a full match.
    return [min(float(s) * 2, 1.0) for s in sims]
