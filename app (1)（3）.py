import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Student Risk Prediction Dashboard", layout="wide")

FEATURES = ["Attendance", "Internal", "Mid-sem", "Assignment", "Study hours"]
LEVELS = ["High", "Medium", "Low"]
COLORS = {"High": "#d1495b", "Medium": "#f4a643", "Low": "#0f7e7e"}

# ---------- styling (black background, yellow tabs) ----------
st.markdown("""
<style>
.stApp{background:#000;color:#fff}
h1{font-family:Georgia,serif;font-weight:700}
button[role=tab]{border:1px solid #ffe600;border-radius:6px;margin-right:8px;padding:6px 14px}
button[role=tab] p{color:#ffe600;font-family:Georgia,serif;font-size:17px}
button[role=tab][aria-selected=true]{background:#ffe600}
button[role=tab][aria-selected=true] p{color:#000}
div[data-baseweb=tab-highlight]{display:none}
[data-testid=stMetric]{border:1px solid #fff;border-radius:10px;padding:14px 18px}
</style>
""", unsafe_allow_html=True)


# ---------- data + model ----------
@st.cache_data
def generate(n=300, seed=42):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    df = pd.DataFrame({
        "Attendance": np.clip(75 + 11 * a + rng.normal(0, 5, n), 50, 100),
        "Internal": np.clip(62 + 11 * a + rng.normal(0, 6, n), 30, 90),
        "Mid-sem": np.clip(58 + 11 * a + rng.normal(0, 7, n), 25, 90),
        "Assignment": np.clip(66 + 8 * a + rng.normal(0, 7, n), 35, 95),
        "Study hours": np.clip(3 + 0.9 * a + rng.normal(0, 0.8, n), 0.5, 6),
    }).round(1)
    rank = (a + rng.normal(0, 0.5, n)).argsort().argsort()
    df["Risk"] = np.where(rank < n * 0.3, "High", np.where(rank < n * 0.4, "Medium", "Low"))
    df.insert(0, "ID", [f"S{i + 1:03d}" for i in range(n)])
    return df


@st.cache_resource
def train(df):
    X, y = df[FEATURES], df["Risk"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42).fit(Xtr, ytr)
    pred = model.predict(Xte)
    return model, accuracy_score(yte, pred) * 100, confusion_matrix(yte, pred, labels=LEVELS)


base = generate()
model, acc, cm = train(base)
if "df" not in st.session_state:
    st.session_state.df = base.copy()
df = st.session_state.df


def dark(fig, h=340):
    fig.update_layout(template="plotly_dark", paper_bgcolor="#000", plot_bgcolor="#000",
                      height=h, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
    return fig


# ---------- header ----------
st.title("Student Risk Prediction Dashboard")
st.caption("Random Forest classifier (scikit-learn) | predicts at-risk students from scores, attendance and study habits")
t1, t2, t3, t4 = st.tabs(["Overview", "Add / Predict", "Model", "Students"])

# ---------- Overview ----------
with t1:
    c = st.columns(4)
    c[0].metric("Students", len(df))
    c[1].metric("High risk", int((df.Risk == "High").sum()))
    c[2].metric("Avg attendance", f"{df.Attendance.mean():.1f}%")
    c[3].metric("Model accuracy", f"{acc:.1f}%")

    g1, g2, g3 = st.columns(3)
    with g1:
        st.subheader("Risk level distribution")
        counts = df.Risk.value_counts().reindex(LEVELS).reset_index()
        counts.columns = ["Risk", "Count"]
        fig = px.pie(counts, names="Risk", values="Count", hole=0.5,
                     color="Risk", color_discrete_map=COLORS)
        fig.update_traces(marker_line_color="#fff", marker_line_width=2)
        st.plotly_chart(dark(fig), use_container_width=True)
    with g2:
        st.subheader("Attendance vs internal marks")
        fig = px.scatter(df, x="Attendance", y="Internal", color="Risk",
                         color_discrete_map=COLORS, category_orders={"Risk": LEVELS})
        st.plotly_chart(dark(fig), use_container_width=True)
    with g3:
        st.subheader("Average by risk level")
        avg = (df.groupby("Risk")[["Attendance", "Internal", "Mid-sem", "Assignment"]]
                 .mean().reindex(LEVELS).reset_index().melt(id_vars="Risk"))
        fig = px.bar(avg, x="variable", y="value", color="Risk", barmode="group",
                     color_discrete_map=COLORS, labels={"variable": "", "value": ""})
        st.plotly_chart(dark(fig), use_container_width=True)

# ---------- Add / Predict ----------
with t2:
    st.subheader("Enter a student's details")
    f = st.columns(5)
    vals = [
        f[0].number_input("Attendance %", 0.0, 100.0, 72.0),
        f[1].number_input("Internal marks %", 0.0, 100.0, 55.0),
        f[2].number_input("Mid-sem marks %", 0.0, 100.0, 50.0),
        f[3].number_input("Assignment marks %", 0.0, 100.0, 60.0),
        f[4].number_input("Study hours per day", 0.0, 12.0, 2.0, step=0.5),
    ]
    if st.button("Predict risk"):
        x = pd.DataFrame([vals], columns=FEATURES)
        proba = dict(zip(model.classes_, model.predict_proba(x)[0]))
        st.session_state.pending = (vals, max(proba, key=proba.get), proba)

    if "pending" in st.session_state:
        vals_p, label, proba = st.session_state.pending
        st.markdown(f"### Predicted risk: <span style='color:{COLORS[label]}'>{label}</span>",
                    unsafe_allow_html=True)
        for lv in LEVELS:
            st.progress(float(proba.get(lv, 0)), text=f"{lv} - {proba.get(lv, 0) * 100:.0f}%")
        if st.button("Add to dataset"):
            new = dict(zip(FEATURES, vals_p))
            new.update({"ID": f"S{len(df) + 1:03d}", "Risk": label})
            st.session_state.df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
            del st.session_state["pending"]
            st.rerun()

# ---------- Model ----------
with t3:
    m1, m2 = st.columns(2)
    with m1:
        st.subheader("Feature importance")
        imp = pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_ * 100})
        fig = px.bar(imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
                     color_discrete_sequence=["#0f7e7e"])
        st.plotly_chart(dark(fig), use_container_width=True)
    with m2:
        st.subheader("Model details")
        st.write(f"Random Forest, {model.n_estimators} trees, max depth {model.max_depth}. "
                 f"Test accuracy: **{acc:.1f}%** (80/20 split).")
        st.subheader("Confusion matrix (test set)")
        fig = px.imshow(cm, x=LEVELS, y=LEVELS, text_auto=True, color_continuous_scale="Teal",
                        labels=dict(x="Predicted", y="Actual"))
        st.plotly_chart(dark(fig, 300), use_container_width=True)

# ---------- Students ----------
with t4:
    pick = st.selectbox("Filter by risk", ["All"] + LEVELS)
    view = df if pick == "All" else df[df.Risk == pick]
    st.dataframe(view[["ID"] + FEATURES + ["Risk"]], use_container_width=True, height=500, hide_index=True)
