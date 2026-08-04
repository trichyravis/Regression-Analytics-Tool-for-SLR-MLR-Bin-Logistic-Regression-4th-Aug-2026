

import io
import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import statsmodels.api as sm
import streamlit as st
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera


st.set_page_config(
    page_title="Regression Analytics Laboratory | Mountain Path Academy",
    page_icon="〽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#0B2545"
NAVY_2 = "#124A78"
GOLD = "#F3C84B"
GOLD_2 = "#D4A017"
BLUE = "#0B5CAD"
GREEN = "#1E9E64"
RED = "#D64545"
PURPLE = "#7B61FF"
CREAM = "#FFF9E8"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
.stApp {{ background:linear-gradient(180deg,#F8FAFD,#EAF1F7); color:#172B3A; }}
html, body, [class*="css"] {{ font-family:'Inter',sans-serif; }}
[data-testid="stSidebar"] {{ background:linear-gradient(180deg,#081F3A,#124A78); color:#F7FAFC; }}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,[data-testid="stSidebar"] p,[data-testid="stSidebar"] li {{ color:#F7FAFC; }}
section[data-testid="stSidebar"] label p {{ color:#F3C84B!important; font-weight:850!important; }}
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] > div {{ background:#FFFFFF !important; color:#102A43 !important; }}
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span,
[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] span {{ color:#102A43 !important; }}
.hero {{ background:linear-gradient(120deg,#071A2F 0%,#0B3B67 58%,#A97908 145%); padding:30px 34px; border-radius:22px; color:white; box-shadow:0 14px 34px rgba(7,26,47,.22); margin-bottom:16px; border:1px solid rgba(243,200,75,.35); }}
.hero h1 {{ font-size:2.25rem; margin:0 0 8px; color:white; font-weight:900; }}
.hero p {{ margin:0; color:#DDEAF4; line-height:1.55; }}
.eyebrow {{ color:#F3C84B; text-transform:uppercase; letter-spacing:.14em; font-weight:900; font-size:.76rem; margin-bottom:.55rem; }}
.concept-card {{ background:white; border:1px solid #D9E5EF; border-top:5px solid #0B5CAD; padding:17px 18px; border-radius:15px; box-shadow:0 5px 16px rgba(18,54,84,.07); min-height:150px; }}
.concept-card h3 {{ color:#0B2545; font-size:1.05rem; margin:0 0 7px; }}
.concept-card p {{ color:#3C5368; font-size:.91rem; line-height:1.5; margin:0; }}
.model-card {{ background:#FFFFFF; border:1px solid #D9E5EF; border-left:6px solid #0B5CAD; padding:18px 20px; border-radius:14px; box-shadow:0 5px 16px rgba(18,54,84,.06); min-height:205px; }}
.model-card h3 {{ color:#0B2545; margin:0 0 8px; }}
.model-card p,.model-card li {{ color:#3C5368; font-size:.91rem; line-height:1.5; }}
.section-title {{ color:#0B2545; font-weight:900; margin-top:1.2rem; }}
.callout {{ background:{CREAM}; border-left:6px solid {GOLD}; border-radius:10px; padding:.9rem 1rem; margin:.7rem 0; }}
.teaching-note {{ background:#EAF7F5; border-left:5px solid #13A89E; padding:13px 16px; border-radius:10px; color:#153C3A; margin:10px 0; }}
.good {{ background:#E9F8F0; border-left:6px solid {GREEN}; border-radius:10px; padding:.8rem 1rem; }}
.warn {{ background:#FFF0F0; border-left:6px solid {RED}; border-radius:10px; padding:.8rem 1rem; }}
div[data-testid="stMetric"] {{ background:white; border:1px solid #DDE8F1; border-top:4px solid {GOLD}; padding:12px; border-radius:13px; box-shadow:0 5px 15px rgba(20,55,85,.06); }}
.stTabs [data-baseweb="tab-list"] {{ gap:9px!important; flex-wrap:wrap!important; background:#D8E3ED!important; padding:10px!important; border-radius:14px!important; }}
.stTabs button[data-baseweb="tab"] {{ flex:1 1 130px!important; min-height:50px!important; background:#0B2545!important; border:2px solid #F3C84B!important; border-radius:10px!important; color:#F3C84B!important; }}
.stTabs button[data-baseweb="tab"] p {{ color:#F3C84B!important; font-weight:850!important; }}
.stTabs button[data-baseweb="tab"][aria-selected="true"] {{ background:linear-gradient(135deg,#F3C84B,#D4A017)!important; }}
.stTabs button[data-baseweb="tab"][aria-selected="true"] p {{ color:#071A2F!important; }}
.stButton button,.stDownloadButton button {{ background:#0B3B67!important; color:white!important; border-radius:10px!important; font-weight:800!important; }}
.profile-card {{ margin-top:14px; padding:15px; border:1px solid rgba(243,200,75,.5); border-radius:14px; background:rgba(255,255,255,.07); }}
.profile-card .profile-name {{ color:#F3C84B; font-size:1rem; font-weight:900; margin-bottom:5px; }}
.profile-card .profile-role {{ color:#F7FAFC; font-size:.78rem; line-height:1.45; }}
.profile-card .profile-links {{ margin-top:10px; }}
.profile-card a {{ color:#F3C84B!important; font-size:.78rem; font-weight:800; text-decoration:none; }}
.profile-card a:hover {{ text-decoration:underline; }}
.footer {{ background:linear-gradient(115deg,#081F3A,#124A78); color:#E6F1F8; padding:22px; border-radius:16px; margin-top:28px; text-align:center; border-top:4px solid #F3C84B; }}
.footer a {{ color:#F3C84B!important; font-weight:800; text-decoration:none; }}

/* High-visibility sidebar controls across current Streamlit/BaseWeb versions */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-baseweb="base-input"],
section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"],
section[data-testid="stSidebar"] [data-baseweb="tag"] {{ background-color:#FFFFFF!important; }}
section[data-testid="stSidebar"] div[data-baseweb="select"] *,
section[data-testid="stSidebar"] div[data-baseweb="base-input"] *,
section[data-testid="stSidebar"] div[data-baseweb="popover"] *,
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {{ color:#102A43!important; -webkit-text-fill-color:#102A43!important; opacity:1!important; }}
section[data-testid="stSidebar"] div[data-baseweb="select"] svg {{ fill:#102A43!important; color:#102A43!important; }}
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"] *,
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] span,
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] small {{ color:#102A43!important; -webkit-text-fill-color:#102A43!important; font-weight:700!important; }}
section[data-testid="stSidebar"] [data-baseweb="tag"] span {{ color:#FFFFFF!important; -webkit-text-fill-color:#FFFFFF!important; }}
section[data-testid="stSidebar"] [data-baseweb="tag"] {{ background:#E45756!important; border-radius:7px!important; }}

/* Keep every file-uploader instruction and button label readable */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {{
  background:linear-gradient(135deg,#F3C84B,#D4A017)!important;
  border:1px solid #FFF1A8!important; color:#071A2F!important;
  min-width:120px!important; font-weight:900!important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button *,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button p {{
  color:#071A2F!important; -webkit-text-fill-color:#071A2F!important; opacity:1!important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] > div > div,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span {{
  color:#40566A!important; -webkit-text-fill-color:#40566A!important; opacity:1!important;
}}

/* Make the primary analysis action impossible to miss */
section[data-testid="stSidebar"] .stButton > button,
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {{
  background:linear-gradient(135deg,#F3C84B,#D4A017)!important;
  color:#071A2F!important; border:2px solid #FFF1A8!important;
  min-height:52px!important; font-size:1rem!important; font-weight:900!important;
  box-shadow:0 8px 20px rgba(0,0,0,.28)!important;
}}
section[data-testid="stSidebar"] .stButton > button p,
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] p {{ color:#071A2F!important; font-weight:900!important; }}
section[data-testid="stSidebar"] .stButton > button:hover {{ background:linear-gradient(135deg,#FFE27A,#F3C84B)!important; transform:translateY(-1px); }}

/* Prominent, responsive result-navigation cards */
div[data-testid="stTabs"] div[data-baseweb="tab-list"],
.stTabs div[role="tablist"] {{
  gap:10px!important; flex-wrap:wrap!important; background:#D8E3ED!important;
  padding:11px!important; border-radius:15px!important; border:1px solid #C3D2DE!important;
}}
div[data-testid="stTabs"] button[role="tab"],
.stTabs button[data-baseweb="tab"] {{
  flex:1 1 145px!important; min-height:54px!important; padding:10px 14px!important;
  background:linear-gradient(145deg,#0B2545,#124A78)!important;
  border:2px solid #F3C84B!important; border-radius:11px!important;
  box-shadow:0 4px 10px rgba(11,37,69,.17)!important;
}}
div[data-testid="stTabs"] button[role="tab"] p,
div[data-testid="stTabs"] button[role="tab"] span,
.stTabs button[data-baseweb="tab"] p {{ color:#F8D967!important; font-weight:900!important; font-size:.92rem!important; }}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
.stTabs button[data-baseweb="tab"][aria-selected="true"] {{
  background:linear-gradient(135deg,#F3C84B,#D4A017)!important;
  border-color:#0B2545!important; box-shadow:0 5px 13px rgba(212,160,23,.35)!important;
}}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p,
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] span,
.stTabs button[data-baseweb="tab"][aria-selected="true"] p {{ color:#071A2F!important; }}
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ display:none!important; }}
@media(max-width:800px) {{ div[data-testid="stTabs"] button[role="tab"] {{ flex:1 1 46%!important; }} }}
</style>
""",
    unsafe_allow_html=True,
)


def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() or f"Column_{i+1}" for i, c in enumerate(df.columns)]
    if df.columns.duplicated().any():
        seen, new = {}, []
        for col in df.columns:
            seen[col] = seen.get(col, 0) + 1
            new.append(col if seen[col] == 1 else f"{col}_{seen[col]}")
        df.columns = new
    return df.dropna(how="all").dropna(axis=1, how="all")


def smart_numeric(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.strip()
    percent = s.str.endswith("%")
    negative = s.str.match(r"^\(.*\)$")
    s = s.str.replace(r"[₹$€£,%]", "", regex=True)
    s = s.str.replace(",", "", regex=False).str.replace(r"[()]", "", regex=True)
    out = pd.to_numeric(s, errors="coerce")
    out.loc[negative & out.notna()] *= -1
    out.loc[percent & out.notna()] /= 100
    return out


@st.cache_data(show_spinner=False)
def read_file(file_bytes, filename):
    stream = io.BytesIO(file_bytes)
    ext = filename.lower().split(".")[-1]
    if ext == "csv":
        return {"CSV Data": clean_columns(pd.read_csv(stream))}
    return {name: clean_columns(df) for name, df in
            pd.read_excel(stream, sheet_name=None, engine="openpyxl").items()}


def numeric_columns(df):
    return [c for c in df.columns if smart_numeric(df[c]).notna().sum() >= max(10, int(.60*len(df)))]


def vif_table(X):
    if X.shape[1] == 1:
        return pd.DataFrame({"Variable": X.columns, "VIF": [1.0], "Interpretation": ["Not applicable with one X"]})
    vals = []
    for i, col in enumerate(X.columns):
        try: value = variance_inflation_factor(X.to_numpy(), i)
        except Exception: value = np.nan
        label = "Low" if value < 5 else "Moderate – review" if value < 10 else "High – investigate"
        vals.append((col, value, label))
    return pd.DataFrame(vals, columns=["Variable", "VIF", "Interpretation"])


def styled_table(df, formats=None):
    styler = df.style
    if formats: styler = styler.format(formats, na_rep="—")
    return styler.set_properties(**{"text-align":"left"}).set_table_styles([
        {"selector":"th", "props":[("background-color",NAVY),("color","white"),("font-weight","700")]},
        {"selector":"td", "props":[("border-bottom","1px solid #E5ECF2")]},
    ])


def prepare_linear(df, y_col, x_cols, missing):
    work = pd.DataFrame({c: smart_numeric(df[c]) for c in [y_col]+x_cols})
    before = len(work)
    if missing == "Remove incomplete rows": work = work.dropna()
    else:
        for c in work.columns: work[c] = work[c].fillna(work[c].median())
    if len(work) < max(20, 5*len(x_cols)+10):
        raise ValueError(f"Only {len(work)} usable rows remain. Review missing/non-numeric values or reduce X variables.")
    if work[y_col].nunique() < 3: raise ValueError("Linear-regression Y must contain at least three distinct numeric values.")
    return work, before-len(work)


def fit_linear(work, y_col, x_cols, confidence):
    X0 = work[x_cols]; X = sm.add_constant(X0, has_constant="add"); y = work[y_col]
    model = sm.OLS(y, X).fit(); ci = model.conf_int(alpha=1-confidence)
    coeff = pd.DataFrame({"Variable":model.params.index,"Coefficient":model.params.values,
        "Std. Error":model.bse.values,"t-statistic":model.tvalues.values,"p-value":model.pvalues.values,
        "CI Lower":ci.iloc[:,0].values,"CI Upper":ci.iloc[:,1].values})
    coeff["Significant at 5%?"] = np.where(coeff["p-value"]<.05,"Yes","No")
    pred = model.get_prediction(X).summary_frame(alpha=1-confidence)
    out = work.copy(); out["Predicted Y"]=model.fittedvalues; out["Residual"]=model.resid
    inf=model.get_influence(); out["Standardized Residual"]=inf.resid_studentized_internal
    out["Mean CI Lower"]=pred["mean_ci_lower"]; out["Mean CI Upper"]=pred["mean_ci_upper"]
    out["Prediction Lower"]=pred["obs_ci_lower"]; out["Prediction Upper"]=pred["obs_ci_upper"]
    out["Leverage"]=inf.hat_matrix_diag; out["Cook's Distance"]=inf.cooks_distance[0]
    jb=jarque_bera(model.resid); bp=het_breuschpagan(model.resid,X)
    try: white=het_white(model.resid,X); reset=linear_reset(model,power=2,use_f=True)
    except Exception: white=(np.nan,np.nan); reset=type("R",(),{"fvalue":np.nan,"pvalue":np.nan})()
    diag=pd.DataFrame([
        ("Durbin–Watson",durbin_watson(model.resid),np.nan,"Near 2 is desirable"),
        ("Jarque–Bera normality",jb[0],jb[1],"p > 0.05: normality not rejected"),
        ("Breusch–Pagan",bp[0],bp[1],"p > 0.05: constant variance not rejected"),
        ("White test",white[0],white[1],"p > 0.05: constant variance not rejected"),
        ("Ramsey RESET",float(reset.fvalue),float(reset.pvalue),"p > 0.05: no strong misspecification evidence"),
    ],columns=["Diagnostic","Statistic","p-value","Guidance"])
    return model, coeff, out, vif_table(X0), diag


def prepare_logistic(df, y_col, x_cols, missing, event_label):
    frame = pd.DataFrame({y_col:df[y_col]})
    for c in x_cols: frame[c]=smart_numeric(df[c])
    before=len(frame)
    if missing=="Remove incomplete rows": frame=frame.dropna()
    else:
        frame=frame.dropna(subset=[y_col])
        for c in x_cols: frame[c]=frame[c].fillna(frame[c].median())
    classes=list(pd.unique(frame[y_col]));
    if len(classes)!=2: raise ValueError(f"Logistic Y must have exactly two categories; found {len(classes)}.")
    if event_label not in classes: raise ValueError("Selected event category is unavailable after cleaning.")
    non_event=[x for x in classes if x!=event_label][0]; mapping={non_event:0,event_label:1}
    frame["Binary Y"]=frame[y_col].map(mapping).astype(int)
    minimum=max(20,4*len(x_cols)+8)
    if len(frame)<minimum: raise ValueError(f"Only {len(frame)} usable rows remain; at least {minimum} are needed.")
    if frame["Binary Y"].value_counts().min()<max(5,len(x_cols)+1): raise ValueError("The smaller Y class has too few observations for the selected predictors.")
    return frame,mapping,before-len(frame)


def hosmer_lemeshow(y,p):
    f=pd.DataFrame({"y":np.asarray(y),"p":np.asarray(p)}); g=min(10,max(2,len(f)//10),f.p.nunique())
    f["Risk Group"]=pd.qcut(f.p,q=g,duplicates="drop",labels=False)+1
    tab=f.groupby("Risk Group").agg(Observations=("y","size"),Observed_Events=("y","sum"),Expected_Events=("p","sum"),Mean_Probability=("p","mean")).reset_index()
    tab["Observed_NonEvents"]=tab.Observations-tab.Observed_Events; tab["Expected_NonEvents"]=tab.Observations-tab.Expected_Events
    eps=1e-10; stat=np.sum((tab.Observed_Events-tab.Expected_Events)**2/np.maximum(tab.Expected_Events,eps)+(tab.Observed_NonEvents-tab.Expected_NonEvents)**2/np.maximum(tab.Expected_NonEvents,eps))
    return float(stat),float(stats.chi2.sf(stat,max(1,len(tab)-2))),tab


def fit_logistic(work, y_col, x_cols, confidence, cutoff):
    y=work["Binary Y"]; X0=work[x_cols]; X=sm.add_constant(X0,has_constant="add")
    model=sm.GLM(y,X,family=sm.families.Binomial()).fit(maxiter=200); ci=model.conf_int(alpha=1-confidence)
    coeff=pd.DataFrame({"Variable":model.params.index,"Log-Odds Coefficient":model.params.values,
        "Std. Error":model.bse.values,"z-statistic":model.tvalues.values,"p-value":model.pvalues.values,
        "Odds Ratio":np.exp(np.clip(model.params.values,-700,700)),
        "OR CI Lower":np.exp(np.clip(ci.iloc[:,0].values,-700,700)),"OR CI Upper":np.exp(np.clip(ci.iloc[:,1].values,-700,700))})
    coeff["Significant at 5%?"]=np.where(coeff["p-value"]<.05,"Yes","No")
    p=np.asarray(model.predict(X)); pred=(p>=cutoff).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    specificity=tn/(tn+fp) if tn+fp else np.nan; auc=roc_auc_score(y,p); hl,hl_p,cal=hosmer_lemeshow(y,p)
    metrics=pd.DataFrame([("Observations",len(y)),("Event rate",y.mean()),("Cutoff",cutoff),
        ("Accuracy",accuracy_score(y,pred)),("Sensitivity / Recall",recall_score(y,pred,zero_division=0)),
        ("Specificity",specificity),("Precision",precision_score(y,pred,zero_division=0)),
        ("F1 Score",f1_score(y,pred,zero_division=0)),("ROC AUC",auc),
        ("Average Precision",average_precision_score(y,p)),("Log-Likelihood",model.llf),("AIC",model.aic),
        ("BIC",-2*model.llf+len(model.params)*np.log(len(y))),("Hosmer–Lemeshow Statistic",hl),("Hosmer–Lemeshow p-value",hl_p)],columns=["Metric","Value"])
    inf=model.get_influence(observed=True); out=work.copy(); out["Predicted Probability"]=p; out["Predicted Class"]=pred
    out["Correct?"]=np.where(pred==y,"Yes","No"); out["Deviance Residual"]=model.resid_deviance
    out["Leverage"]=inf.hat_matrix_diag; out["Cook's Distance"]=inf.cooks_distance[0]
    matrix=pd.DataFrame([[tn,fp],[fn,tp]],index=["Actual 0","Actual 1"],columns=["Predicted 0","Predicted 1"])
    return model,coeff,out,vif_table(X0),metrics,matrix,cal


def excel_report(model_name, source, sheet, tables, metadata):
    output=io.BytesIO()
    with pd.ExcelWriter(output,engine="xlsxwriter",datetime_format="dd-mmm-yyyy") as writer:
        wb=writer.book; title=wb.add_format({"bg_color":NAVY,"font_color":"white","bold":True,"font_size":18,"align":"left"})
        header=wb.add_format({"bg_color":NAVY,"font_color":"white","bold":True,"align":"center","text_wrap":True})
        label=wb.add_format({"bg_color":"#EAF2F8","font_color":NAVY,"bold":True}); num=wb.add_format({"num_format":"0.0000"})
        dash=wb.add_worksheet("Dashboard"); writer.sheets["Dashboard"]=dash; dash.hide_gridlines(2); dash.set_column("A:A",30); dash.set_column("B:B",42)
        dash.merge_range("A1:H1",f"MOUNTAIN PATH ACADEMY – {model_name.upper()}",title)
        meta=[("Source File",source),("Worksheet",sheet)]+metadata
        for r,(k,v) in enumerate(meta,start=2): dash.write(r,0,k,label); dash.write(r,1,v,num if isinstance(v,(float,np.floating)) else None)
        dash.write(len(meta)+4,0,"Educational Use",label); dash.merge_range(len(meta)+4,1,len(meta)+5,7,"Association does not establish causation. Validate assumptions, specification, stability and out-of-sample performance before decision use.")
        colors=[BLUE,GREEN,PURPLE,GOLD,RED,"#00ACC1","#607D8B"]
        for i,(name,df) in enumerate(tables.items()):
            safe=name[:31]; df.to_excel(writer,sheet_name=safe,index=True if name=="Confusion Matrix" else False,startrow=2); ws=writer.sheets[safe]
            ws.hide_gridlines(2); ws.freeze_panes(3,1); end=max(1,len(df.columns)-1+(1 if name=="Confusion Matrix" else 0)); ws.merge_range(0,0,0,end,safe.upper(),title); ws.set_row(2,28)
            offset=1 if name=="Confusion Matrix" else 0
            if offset: ws.write(2,0,"Actual Class",header)
            for c,col in enumerate(df.columns,start=offset): ws.write(2,c,str(col),header); ws.set_column(c,c,max(13,min(30,len(str(col))+5)))
            ws.set_tab_color(colors[i%len(colors)])
    output.seek(0); return output.getvalue()


def plot_linear(work,y_col,x_cols,model,pred):
    if len(x_cols)==1:
        x=x_cols[0]; d=pred.sort_values(x); fig=go.Figure()
        fig.add_trace(go.Scatter(x=work[x],y=work[y_col],mode="markers",name="Actual",marker=dict(color=BLUE,size=8)))
        fig.add_trace(go.Scatter(x=d[x],y=d["Predicted Y"],mode="lines",name="Regression line",line=dict(color=RED,width=3)))
        fig.update_layout(title=f"{y_col} explained by {x}",xaxis_title=x,yaxis_title=y_col)
    else:
        fig=px.scatter(pred,x="Predicted Y",y=y_col,title=f"Actual versus Predicted {y_col}",color_discrete_sequence=[BLUE])
        lo=min(pred["Predicted Y"].min(),pred[y_col].min()); hi=max(pred["Predicted Y"].max(),pred[y_col].max()); fig.add_shape(type="line",x0=lo,y0=lo,x1=hi,y1=hi,line=dict(color=RED,dash="dash"))
    return fig


def profile_block():
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
<div class="profile-card">
  <div class="profile-name">Prof. V. Ravichandran</div>
  <div class="profile-role">Faculty · Finance, Risk &amp; Quantitative Analytics<br>28+ years in corporate finance and banking · 10+ years in academia</div>
  <div class="profile-links">
    <a href="https://www.linkedin.com/in/trichyravis" target="_blank">LinkedIn ↗</a> &nbsp;·&nbsp;
    <a href="https://github.com/trichyravis" target="_blank">GitHub ↗</a><br>
    <a href="https://themountainpathacademy.com/about.html" target="_blank">Full faculty profile ↗</a>
    <br><br><strong>Mountain Path Academy</strong><br>
    <a href="https://themountainpathacademy.com/courses" target="_blank">Explore courses</a> &nbsp;·&nbsp;
    <a href="https://themountainpathacademy.com/contact" target="_blank">Contact &amp; enrol</a>
  </div>
</div>""",unsafe_allow_html=True)


def render_footer():
    st.markdown("""
<div class='footer'>
  <strong>Prof. V. Ravichandran · The Mountain Path Academy</strong><br>
  Visiting Faculty — NMIMS Bangalore · BITS Pilani (WILP) · RV University Bangalore · Goa Institute of Management<br>
  28+ years in corporate finance and banking · Finance, Risk Management &amp; Quantitative Analytics<br><br>
  <a href='https://www.linkedin.com/in/trichyravis' target='_blank'>LinkedIn</a> &nbsp;·&nbsp;
  <a href='https://github.com/trichyravis' target='_blank'>GitHub</a> &nbsp;·&nbsp;
  <a href='https://themountainpathacademy.com/about.html' target='_blank'>Full faculty profile</a>
  <br><br><strong>Mountain Path Academy</strong> &nbsp;
  <a href='https://themountainpathacademy.com/courses' target='_blank'><strong>Explore courses</strong></a> &nbsp;·&nbsp;
  <a href='https://themountainpathacademy.com/contact' target='_blank'><strong>Contact &amp; enrol</strong></a><br>
  <small>Regression Analytics Laboratory · Educational material only · Model output is not a substitute for professional judgement</small>
</div>
""",unsafe_allow_html=True)


def render_educational_tool():
    st.markdown("## 📚 Regression Educational Tool")
    st.markdown("""<div class='teaching-note'><b>How to use this section:</b> Begin with the business question and the type of dependent variable. Then select the correct model, interpret its coefficients, assess fit and assumptions, and validate performance before using the results.</div>""",unsafe_allow_html=True)
    t1,t2,t3=st.tabs(["SLR","MLR","Logistic Regression"])
    with t1:
        st.markdown(r"""### Simple Linear Regression
**Purpose:** Explain or predict one continuous dependent variable using one numeric independent variable.

**Model:**  \(Y = \\beta_0 + \\beta_1X + \\varepsilon\)

- **β₀ (intercept):** expected value of Y when X is zero.
- **β₁ (slope):** expected change in Y for a one-unit increase in X.
- **R²:** proportion of sample variation in Y explained by X.
- **p-value and confidence interval:** evidence about whether the population slope differs from zero.
- **Check:** linearity, independent errors, constant residual variance, residual normality and influential observations.

**Example:** Estimate how advertising expenditure is associated with sales revenue.""")
    with t2:
        st.markdown(r"""### Multiple Linear Regression
**Purpose:** Explain or predict one continuous dependent variable using two or more numeric independent variables.

**Model:**  \(Y = \\beta_0 + \\beta_1X_1 + \\cdots + \\beta_kX_k + \\varepsilon\)

- Interpret each coefficient as the expected change in Y for a one-unit change in that X, **holding the other included X variables constant**.
- Use **adjusted R²** when comparing models with different numbers of predictors.
- Use the overall **F-test** to assess whether the predictors jointly explain Y.
- Review **VIF** for multicollinearity and residual plots for specification problems.
- Statistical significance does not by itself establish causation or practical importance.

**Example:** Explain firm profitability using sales growth, leverage and asset turnover.""")
    with t3:
        st.markdown(r"""### Binary Logistic Regression
**Purpose:** Estimate the probability of an outcome with two classes, such as default/no default.

**Model:**  \(\\log[p/(1-p)] = \\beta_0 + \\beta_1X_1 + \\cdots + \\beta_kX_k\)

- A coefficient describes a change in **log-odds**; `exp(β)` is the **odds ratio**.
- Odds ratio above 1 increases estimated event odds; below 1 decreases them, holding other predictors constant.
- **ROC AUC** evaluates ranking ability across classification cutoffs.
- **Sensitivity** measures event detection; **specificity** measures non-event detection.
- Accuracy depends on the cutoff and class balance; inspect the confusion matrix and calibration.

**Example:** Estimate probability of loan default using income, debt burden and credit history.""")
    st.markdown("""<div class='callout'><b>Responsible interpretation:</b> Regression identifies conditional associations in the analysed sample. Always consider data quality, omitted variables, functional form, sample size, stability and out-of-sample validation. Model output supports judgement; it does not replace it.</div>""",unsafe_allow_html=True)


st.markdown("""<div class="hero"><div class="eyebrow">Mountain Path Academy · Global Financial Analytics Studio</div><h1>Regression Analytics Laboratory</h1><p>Simple Linear Regression · Multiple Linear Regression · Binary Logistic Regression</p><p>Use your own data to estimate relationships, test statistical significance, validate model assumptions and translate regression output into practical decision insights.</p></div>""",unsafe_allow_html=True)

st.sidebar.markdown("## 〽️ Mountain Path Academy")
st.sidebar.caption("Global regression analytics laboratory")
st.sidebar.markdown("### Model Control Centre")
uploaded=st.sidebar.file_uploader("Upload Excel or CSV",type=["xlsx","xlsm","csv"])

if uploaded is None:
    profile_block()
    c1,c2,c3=st.columns(3)
    with c1: st.markdown("<div class='concept-card'><h3>① Upload your data</h3><p>Use an Excel or CSV dataset. Multiple Excel worksheets are supported.</p></div>",unsafe_allow_html=True)
    with c2: st.markdown("<div class='concept-card'><h3>② Select a model</h3><p>Choose SLR or MLR for a continuous outcome, or logistic regression for a binary outcome.</p></div>",unsafe_allow_html=True)
    with c3: st.markdown("<div class='concept-card'><h3>③ Interpret and export</h3><p>Review coefficients, diagnostics, predictions and learning notes, then download a formatted Excel report.</p></div>",unsafe_allow_html=True)

    st.markdown("<h2 class='section-title'>About this educational tool</h2>",unsafe_allow_html=True)
    st.markdown("""<div class='callout'><b>Purpose:</b> This interactive laboratory connects regression theory with hands-on analysis. It guides learners from variable selection and estimation to inference, diagnostics, prediction and responsible interpretation. It is designed for classroom demonstrations, guided practice and exploratory analysis—not as an automated decision engine.</div>""",unsafe_allow_html=True)
    a1,a2,a3=st.columns(3)
    with a1:
        st.markdown("""<div class='model-card'><h3>Simple Linear Regression (SLR)</h3><p><b>Use when:</b> one numeric X explains or predicts one continuous Y.</p><p><b>Model:</b> Y = β₀ + β₁X + ε</p><ul><li>Interpret slope and intercept</li><li>Review R², p-values and confidence intervals</li><li>Check residual linearity, normality and constant variance</li></ul></div>""",unsafe_allow_html=True)
    with a2:
        st.markdown("""<div class='model-card'><h3>Multiple Linear Regression (MLR)</h3><p><b>Use when:</b> several numeric X variables explain or predict one continuous Y.</p><p><b>Model:</b> Y = β₀ + β₁X₁ + … + βₖXₖ + ε</p><ul><li>Interpret each coefficient holding other X variables constant</li><li>Compare R² and adjusted R²</li><li>Check VIF, residual diagnostics and specification</li></ul></div>""",unsafe_allow_html=True)
    with a3:
        st.markdown("""<div class='model-card'><h3>Binary Logistic Regression</h3><p><b>Use when:</b> Y has two outcomes such as default/no default.</p><p><b>Model:</b> log[p/(1−p)] = β₀ + β₁X₁ + … + βₖXₖ</p><ul><li>Interpret odds ratios and predicted probabilities</li><li>Review ROC AUC, sensitivity and specificity</li><li>Study cutoff effects, calibration and class balance</li></ul></div>""",unsafe_allow_html=True)

    st.markdown("""<div class='teaching-note'><b>Learning workflow:</b> Frame the question → choose Y and X → estimate the model → assess statistical significance → test assumptions and fit → interpret practical meaning → validate before decision use.</div>""",unsafe_allow_html=True)
    st.info("Upload a dataset from the left sidebar to begin.")
    render_footer()
    st.stop()

try: sheets=read_file(uploaded.getvalue(),uploaded.name)
except Exception as e: st.error(f"Could not read the file: {e}"); st.stop()

sheet=st.sidebar.selectbox("Worksheet",list(sheets)); df=sheets[sheet]
model_choice=st.sidebar.radio("Regression model",["Linear Regression — SLR/MLR","Binary Logistic Regression"])
nums=numeric_columns(df)
if len(nums)<1: st.error("No substantially numeric X variables were detected."); st.stop()

if model_choice.startswith("Linear"):
    y_options=nums
else:
    y_options=[c for c in df.columns if 2 <= df[c].dropna().nunique() <= 20]
    if not y_options: st.error("No suitable categorical Y variable was detected."); st.stop()
y_col=st.sidebar.selectbox("Dependent variable (Y)",y_options)
x_options=[c for c in nums if c!=y_col]
x_cols=st.sidebar.multiselect("Independent variable(s) (X)",x_options,default=x_options[:1])
missing=st.sidebar.selectbox("Missing-data treatment",["Remove incomplete rows","Median-impute numeric X/Y"])
confidence=st.sidebar.select_slider("Confidence level",options=[.90,.95,.99],value=.95,format_func=lambda x:f"{x:.0%}")

event_label=None; cutoff=.50
if model_choice.startswith("Binary"):
    events=list(pd.unique(df[y_col].dropna())); event_label=st.sidebar.selectbox("Event category coded 1",events,index=len(events)-1)
    cutoff=st.sidebar.slider("Classification cutoff",.10,.90,.50,.05)
run=st.sidebar.button("Run Regression Analysis",use_container_width=True)
profile_block()

st.markdown("### Dataset overview")
m1,m2,m3,m4=st.columns(4); m1.metric("Rows",f"{len(df):,}"); m2.metric("Columns",len(df.columns)); m3.metric("Numeric candidates",len(nums)); m4.metric("Missing cells",f"{int(df.isna().sum().sum()):,}")
with st.expander("Preview uploaded data",expanded=False): st.dataframe(df.head(100),use_container_width=True,height=330)

if not run:
    st.markdown("<div class='callout'><b>Ready:</b> choose the model and variables in the Model Control Centre, then click <b>Run Regression Analysis</b>.</div>",unsafe_allow_html=True)
    render_footer()
    st.stop()
if not x_cols: st.error("Select at least one X variable."); st.stop()

try:
    if model_choice.startswith("Linear"):
        work,dropped=prepare_linear(df,y_col,x_cols,missing); model,coeff,pred,vif,diag=fit_linear(work,y_col,x_cols,confidence)
        model_name="Simple Linear Regression" if len(x_cols)==1 else "Multiple Linear Regression"
        rmse=math.sqrt(np.mean(model.resid**2)); mae=np.mean(np.abs(model.resid))
        st.markdown(f"## {model_name} results")
        c=st.columns(6); c[0].metric("Observations",int(model.nobs)); c[1].metric("R²",f"{model.rsquared:.4f}"); c[2].metric("Adjusted R²",f"{model.rsquared_adj:.4f}"); c[3].metric("Model p-value",f"{model.f_pvalue:.4g}"); c[4].metric("RMSE",f"{rmse:.4f}"); c[5].metric("MAE",f"{mae:.4f}")
        tabs=st.tabs(["📌 Executive Summary","β Coefficients","📊 Visual Analysis","🩺 Diagnostics","🎯 Predictions","🎓 Learning Corner","📚 Educational Tool","📥 Excel Report"])
        with tabs[0]:
            verdict="statistically significant" if model.f_pvalue<.05 else "not statistically significant at 5%"
            st.markdown(f"<div class='good'><b>Overall model:</b> {verdict}. The selected X variables explain <b>{model.rsquared:.1%}</b> of the sample variation in {y_col}.</div>",unsafe_allow_html=True)
            st.markdown(f"**Estimated equation:**  Ŷ = {model.params.iloc[0]:.4f} " + " ".join([f"{model.params[x]:+.4f}({x})" for x in x_cols]))
            st.dataframe(styled_table(pd.DataFrame({"Metric":["AIC","BIC","F-statistic","Rows removed/imputed"],"Value":[model.aic,model.bic,model.fvalue,dropped]}),{"Value":"{:.4f}"}),use_container_width=True)
        with tabs[1]: st.dataframe(styled_table(coeff,{"Coefficient":"{:.6f}","Std. Error":"{:.6f}","t-statistic":"{:.4f}","p-value":"{:.6f}","CI Lower":"{:.6f}","CI Upper":"{:.6f}"}),use_container_width=True)
        with tabs[2]:
            st.plotly_chart(plot_linear(work,y_col,x_cols,model,pred),use_container_width=True)
            c1,c2=st.columns(2)
            c1.plotly_chart(px.scatter(pred,x="Predicted Y",y="Residual",title="Residuals versus Fitted",color_discrete_sequence=[PURPLE]),use_container_width=True)
            qq=stats.probplot(pred["Standardized Residual"],dist="norm"); qfig=go.Figure(go.Scatter(x=qq[0][0],y=qq[0][1],mode="markers",marker_color=BLUE)); qfig.add_trace(go.Scatter(x=qq[0][0],y=qq[1][1]+qq[1][0]*qq[0][0],mode="lines",line_color=RED)); qfig.update_layout(title="Normal Q–Q Plot",xaxis_title="Theoretical quantiles",yaxis_title="Ordered residuals"); c2.plotly_chart(qfig,use_container_width=True)
            st.plotly_chart(px.imshow(work[[y_col]+x_cols].corr(),text_auto=".2f",color_continuous_scale="RdBu_r",zmin=-1,zmax=1,title="Correlation Matrix"),use_container_width=True)
        with tabs[3]: st.markdown("#### Assumption tests"); st.dataframe(styled_table(diag,{"Statistic":"{:.4f}","p-value":"{:.6f}"}),use_container_width=True); st.markdown("#### Variance Inflation Factors"); st.dataframe(styled_table(vif,{"VIF":"{:.3f}"}),use_container_width=True)
        with tabs[4]: st.dataframe(pred,use_container_width=True,height=430)
        with tabs[5]: st.markdown("""<div class='callout'><b>Interpretation discipline</b><br>R² measures sample fit, not causality. A significant coefficient describes the expected change in Y for a one-unit change in X, holding the other included X variables constant. Always review linearity, independence, constant variance, residual normality, multicollinearity and influential observations.</div>""",unsafe_allow_html=True)
        with tabs[6]: render_educational_tool()
        report=excel_report(model_name,uploaded.name,sheet,{"Analysis Data":work,"Coefficients":coeff,"Diagnostics":diag,"VIF":vif,"Predictions":pred},[("Y Variable",y_col),("X Variables",", ".join(x_cols)),("R-squared",model.rsquared),("Adjusted R-squared",model.rsquared_adj),("Model p-value",model.f_pvalue),("RMSE",rmse)])
        with tabs[7]: st.download_button("⬇ Download formatted Excel report",report,f"MP1_{model_name.replace(' ','_')}_{datetime.now():%Y%m%d}.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)
    else:
        work,mapping,dropped=prepare_logistic(df,y_col,x_cols,missing,event_label); model,coeff,pred,vif,metrics,matrix,cal=fit_logistic(work,y_col,x_cols,confidence,cutoff)
        mv=metrics.set_index("Metric")["Value"]
        st.markdown("## Binary Logistic Regression results")
        c=st.columns(6); c[0].metric("Observations",int(mv["Observations"])); c[1].metric("ROC AUC",f"{mv['ROC AUC']:.4f}"); c[2].metric("Accuracy",f"{mv['Accuracy']:.1%}"); c[3].metric("Sensitivity",f"{mv['Sensitivity / Recall']:.1%}"); c[4].metric("Specificity",f"{mv['Specificity']:.1%}"); c[5].metric("F1 Score",f"{mv['F1 Score']:.4f}")
        tabs=st.tabs(["📌 Executive Summary","β Odds Ratios","📈 ROC & Classification","🩺 Diagnostics","🎯 Predictions","🎓 Learning Corner","📚 Educational Tool","📥 Excel Report"])
        with tabs[0]:
            st.markdown(f"<div class='good'><b>Event definition:</b> {event_label} is coded 1. ROC AUC is <b>{mv['ROC AUC']:.3f}</b>; accuracy at the {cutoff:.2f} cutoff is <b>{mv['Accuracy']:.1%}</b>.</div>",unsafe_allow_html=True)
            st.dataframe(styled_table(metrics,{"Value":"{:.6f}"}),use_container_width=True)
        with tabs[1]: st.dataframe(styled_table(coeff,{"Log-Odds Coefficient":"{:.6f}","Std. Error":"{:.6f}","z-statistic":"{:.4f}","p-value":"{:.6f}","Odds Ratio":"{:.4f}","OR CI Lower":"{:.4f}","OR CI Upper":"{:.4f}"}),use_container_width=True)
        with tabs[2]:
            fpr,tpr,_=roc_curve(pred["Binary Y"],pred["Predicted Probability"]); precision,recall,_=precision_recall_curve(pred["Binary Y"],pred["Predicted Probability"])
            c1,c2=st.columns(2); rfig=go.Figure(); rfig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=f"AUC {mv['ROC AUC']:.3f}",line=dict(color=BLUE,width=3))); rfig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line=dict(color="gray",dash="dash"),showlegend=False)); rfig.update_layout(title="ROC Curve",xaxis_title="False Positive Rate",yaxis_title="True Positive Rate"); c1.plotly_chart(rfig,use_container_width=True)
            pfig=go.Figure(go.Scatter(x=recall,y=precision,mode="lines",line=dict(color=GREEN,width=3))); pfig.update_layout(title="Precision–Recall Curve",xaxis_title="Recall",yaxis_title="Precision"); c2.plotly_chart(pfig,use_container_width=True)
            st.markdown("#### Confusion matrix"); st.dataframe(matrix,use_container_width=True)
        with tabs[3]: st.markdown("#### Hosmer–Lemeshow calibration groups"); st.dataframe(cal,use_container_width=True); st.markdown("#### Variance Inflation Factors"); st.dataframe(styled_table(vif,{"VIF":"{:.3f}"}),use_container_width=True); st.plotly_chart(px.scatter(pred,x="Predicted Probability",y="Deviance Residual",title="Deviance Residuals versus Probability",color_discrete_sequence=[PURPLE]),use_container_width=True)
        with tabs[4]: st.dataframe(pred,use_container_width=True,height=430)
        with tabs[5]: st.markdown("""<div class='callout'><b>How to read odds ratios</b><br>An odds ratio above 1 increases estimated event odds; below 1 decreases them, holding other included predictors constant. ROC AUC measures ranking across cutoffs. Accuracy depends on cutoff and class balance. Validate the model on unseen data before decision use.</div>""",unsafe_allow_html=True)
        with tabs[6]: render_educational_tool()
        report=excel_report("Binary Logistic Regression",uploaded.name,sheet,{"Analysis Data":work,"Class Mapping":pd.DataFrame([{"Original Category":k,"Binary Code":v} for k,v in mapping.items()]),"Model Metrics":metrics,"Odds Ratios":coeff,"Confusion Matrix":matrix,"HL Calibration":cal,"VIF":vif,"Predictions":pred},[("Y Variable",y_col),("Event coded 1",str(event_label)),("X Variables",", ".join(x_cols)),("ROC AUC",mv["ROC AUC"]),("Accuracy",mv["Accuracy"]),("Classification cutoff",cutoff)])
        with tabs[7]: st.download_button("⬇ Download formatted Excel report",report,f"MP1_Logistic_Regression_{datetime.now():%Y%m%d}.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)
except Exception as e:
    st.error(f"Analysis could not be completed: {e}")
    st.info("Check that the selected variables contain usable values, Y is appropriate for the chosen model, both logistic classes are represented, and the number of predictors is reasonable for the sample size.")

render_footer()
