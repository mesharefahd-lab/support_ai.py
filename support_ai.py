import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="نظام الدعم الفني الذكي", page_icon="🤖")

st.title("🤖 نظام دعم فني بالذكاء الاصطناعي")
st.write("برنامج بسيط وسهل يحدد نوع المشكلة وأولويتها باستخدام الذكاء الاصطناعي.")

@st.cache_resource
def load_models():
    classifier = pipeline("zero-shot-classification", 
        model="joeddav/xlm-roberta-large-xnli")
    sentiment = pipeline("sentiment-analysis",
        model="CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment")
    return classifier, sentiment

classifier, sentiment = load_models()


categories = [
    "مشكلة شبكة",
    "مشكلة أجهزة",
    "مشكلة برمجيات",
    "طلب صلاحيات",
    "استفسار عام"
]

st.subheader("📝 إنشاء تذكرة دعم فني")

title = st.text_input("عنوان المشكلة")
desc = st.text_area("وصف المشكلة")

if st.button("تحليل التذكرة بالذكاء الاصطناعي"):
    if desc.strip() == "":
        st.warning("الرجاء كتابة وصف المشكلة أولاً")
    else:
        with st.spinner("جاري تحليل التذكرة بالذكاء الاصطناعي..."):
            result = classifier(desc, categories, multi_label=False)
            category = result["labels"][0]
            confidence = result["scores"][0]

            sent = sentiment(desc)[0]
            priority = "عالية" if sent["label"] == "negative" else "منخفضة"

        st.success("✔ تم تحليل التذكرة")

        st.write("**نوع المشكلة:** ", category)
        st.write("**نسبة الثقة:** ", f"{confidence*100:.1f}%")
        st.write("**درجة الأولوية:** ", priority)

        st.info("🎫 تم إنشاء التذكرة (نسخة تجريبية)")
