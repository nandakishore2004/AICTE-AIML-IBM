import streamlit as st
import re
from io import BytesIO
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

pdfmetrics.registerFont(TTFont("Times", "times.ttf"))
pdfmetrics.registerFont(TTFont("Times-Bold", "timesbd.ttf"))

st.set_page_config(page_title="AI Resume Builder", layout="wide")

ACTION_MAP = {
    "did": "Developed",
    "worked": "Implemented",
    "made": "Designed",
    "created": "Built",
    "helped": "Assisted",
    "used": "Utilized"
}

def improve(text):
    if not text.strip():
        return ""
    text = text.lower().strip()
    for k, v in ACTION_MAP.items():
        text = re.sub(rf"\b{k}\b", v.lower(), text)
    text = text.capitalize()
    if not text.endswith("."):
        text += "."
    return text

def language_score(text):
    if not text.strip():
        return 0
    return round(TextBlob(text).sentiment.polarity, 2)

def skill_score(skills):
    if not skills.strip():
        return 0, 0
    corpus = [
        "python java sql machine learning ai data science",
        "communication teamwork leadership problem solving",
        skills
    ]
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(corpus)
    sim = cosine_similarity(vectors[2], vectors[:2])[0]
    return round(sim[0], 2), round(sim[1], 2)

def generate_pdf(data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=40, rightMargin=40,
        topMargin=40, bottomMargin=40
    )

    styles = {
        "name": ParagraphStyle("name", fontName="Times-Bold", fontSize=20, alignment=1, spaceAfter=6),
        "contact": ParagraphStyle("contact", fontName="Times", fontSize=11, alignment=1, spaceAfter=16),
        "heading": ParagraphStyle("heading", fontName="Times-Bold", fontSize=12, spaceBefore=10, spaceAfter=4),
        "text": ParagraphStyle("text", fontName="Times", fontSize=11, spaceAfter=6)
    }

    story = []
    story.append(Paragraph(data["name"].upper(), styles["name"]))
    story.append(Paragraph(f"{data['phone']} | {data['email']}", styles["contact"]))

    def section(t, c):
        if c.strip():
            story.append(Paragraph(t.upper(), styles["heading"]))
            story.append(Paragraph(c, styles["text"]))

    section("Career Objective", data["objective"])
    section("Education", data["education"])
    section("Academic Projects", data["projects"])
    section("Technical Skills", data["skills"])
    section("Certifications", data["certifications"])
    section("Soft Skills", data["soft_skills"])
    section("Achievements", data["achievements"])

    def border(canvas, doc):
        canvas.setStrokeColor(colors.black)
        canvas.rect(25, 25, A4[0]-50, A4[1]-50)

    doc.build(story, onFirstPage=border, onLaterPages=border)
    buffer.seek(0)
    return buffer

left, right = st.columns([1, 1.6])

with left:
    st.subheader("Enter Resume Details")

    name = st.text_input("Full Name")
    phone = st.text_input("Phone Number")
    email = st.text_input("Email")

    objective = st.text_area("Career Objective")

    education = st.text_area("Education (one entry per line)")
    projects = st.text_area("Academic Projects (one project per line)")
    skills = st.text_area("Technical Skills")
    certifications = st.text_area("Certifications (one per line)")
    soft_skills = st.text_area("Soft Skills")
    achievements = st.text_area("Achievements (one per line)")

    generate = st.button("Generate Resume")

with right:
    if generate and name.strip():
        st.markdown(
            f"""
            <div style="text-align:center; margin-bottom:15px;">
                <div style="font-size:28px; font-weight:700;">{name}</div>
                <div style="font-size:14px; color:gray;">{phone} | {email}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### Career Objective")
        st.write(improve(objective))

        st.markdown("### Education")
        for e in education.split("\n"):
            if e.strip():
                st.write("•", improve(e))

        st.markdown("### Academic Projects")
        for p in projects.split("\n"):
            if p.strip():
                st.write("•", improve(p))

        st.markdown("### Technical Skills")
        st.write(skills)

        st.markdown("### Certifications")
        for c in certifications.split("\n"):
            if c.strip():
                st.write("•", c)

        st.markdown("### Soft Skills")
        st.write(soft_skills)

        st.markdown("### Achievements")
        for a in achievements.split("\n"):
            if a.strip():
                st.write("•", improve(a))

        st.markdown("## 🧠 AI Analysis")
        st.write("Language Quality Score:", language_score(objective))
        tech_score, soft_score = skill_score(skills)
        st.write("Technical Skill Relevance:", tech_score)
        st.write("Soft Skill Relevance:", soft_score)

        pdf_data = {
            "name": name,
            "phone": phone,
            "email": email,
            "objective": improve(objective),
            "education": education,
            "projects": projects,
            "skills": skills,
            "certifications": certifications,
            "soft_skills": soft_skills,
            "achievements": achievements
        }

        pdf = generate_pdf(pdf_data)

        st.download_button(
            "⬇️ Download Resume (PDF)",
            data=pdf,
            file_name="AI_Resume.pdf",
            mime="application/pdf"
        )
