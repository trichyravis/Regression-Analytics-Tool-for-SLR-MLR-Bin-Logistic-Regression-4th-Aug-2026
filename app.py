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
    page_title="MP1 Regression Laboratory",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#071E3D"
NAVY_2 = "#0B3158"
GOLD = "#F4B400"
GOLD_2 = "#FFD166"
BLUE = "#2E86DE"
GREEN = "#1E9E64"
RED = "#D64545"
PURPLE = "#7B61FF"
CREAM = "#FFF9E8"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Merriweather:wght@700&display=swap');
.stApp {{ background: linear-gradient(180deg,#F7FAFD 0%,#EEF4F9 100%); color:#172B3A; }}
html, body, [class*="css"] {{ font-family:'Inter',sans-serif; }}
[data-testid="stSidebar"] {{ background:linear-gradient(180deg,{NAVY} 0%,{NAVY_2} 100%); }}
[data-testid="stSidebar"] * {{ color:#FFFFFF !important; }}
[data-testid="stSidebar"] label {{ font-weight:700 !important; color:{GOLD_2} !important; }}
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] > div {{ background:#FFFFFF !important; color:#102A43 !important; }}
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span,
[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] span {{ color:#102A43 !important; }}
.hero {{ background:linear-gradient(120deg,{NAVY} 0%,#124E78 72%,#176B87 100%); padding:2rem 2.2rem; border-radius:20px; color:white; box-shadow:0 14px 35px rgba(7,30,61,.18); border-bottom:5px solid {GOLD}; }}
.hero h1 {{ font-family:'Merriweather',serif; font-size:2.35rem; margin:0; }}
.hero p {{ font-size:1.05rem; margin:.65rem 0 0; color:#E9F4FF; }}
.gold-line {{ width:95px; height:4px; background:{GOLD}; border-radius:5px; margin:.9rem 0; }}
.mp-card {{ background:#FFFFFF; border:1px solid #DCE7F0; border-radius:16px; padding:1rem 1.15rem; box-shadow:0 7px 20px rgba(20,55,85,.07); height:100%; }}
.mp-card h3 {{ color:{NAVY}; margin:.1rem 0 .5rem; }}
.callout {{ background:{CREAM}; border-left:6px solid {GOLD}; border-radius:10px; padding:.9rem 1rem; margin:.7rem 0; }}
.good {{ background:#E9F8F0; border-left:6px solid {GREEN}; border-radius:10px; padding:.8rem 1rem; }}
.warn {{ background:#FFF0F0; border-left:6px solid {RED}; border-radius:10px; padding:.8rem 1rem; }}
div[data-testid="stMetric"] {{ background:white; border:1px solid #DDE8F1; border-top:4px solid {GOLD}; padding:12px; border-radius:13px; box-shadow:0 5px 15px rgba(20,55,85,.06); }}
.stTabs [data-baseweb="tab-list"] {{ gap:8px; flex-wrap:wrap; }}
.stTabs [data-baseweb="tab"] {{ background:{NAVY}; color:{GOLD_2}; border-radius:9px 9px 0 0; padding:10px 16px; font-weight:700; }}
.stTabs [aria-selected="true"] {{ background:{GOLD} !important; color:{NAVY} !important; }}
.stButton > button, .stDownloadButton > button {{ background:{GOLD}; color:{NAVY}; border:0; border-radius:10px; font-weight:800; padding:.65rem 1.2rem; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ background:{GOLD_2}; color:{NAVY}; }}
.profile {{ background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.17); border-radius:14px; padding:1rem; margin-top:1rem; }}
.footer {{ background:{NAVY}; color:white; border-top:5px solid {GOLD}; padding:1.4rem; border-radius:15px; margin-top:2rem; text-align:center; }}
.footer a {{ color:{GOLD_2}; text-decoration:none; font-weight:700; }}
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
    st.sidebar.markdown("### 👨‍🏫 Prof. V. Ravichandran")
    st.sidebar.markdown("""
<div class="profile"><b>Founder — The Mountain Path Academy</b><br><br>
MS (Information Technology) • MBA (Finance)<br><br>
28+ years of industry experience, including wholesale banking risk management; 10+ years of management education and executive learning.<br><br>
<b>Teaching:</b> Financial Risk Management, Derivatives, Investments, Financial Analytics, Valuation and Applied Finance through Excel/Python.
</div>""",unsafe_allow_html=True)
    st.sidebar.link_button("🌐 Mountain Path Academy","https://themountainpathacademy.com/",use_container_width=True)


st.markdown("""<div class="hero"><h1>Regression Analytics Laboratory</h1><div class="gold-line"></div><p>Simple Linear Regression • Multiple Linear Regression • Binary Logistic Regression</p><p>Upload your data, build the model, diagnose assumptions, interpret results and download a complete Excel report.</p></div>""",unsafe_allow_html=True)

st.sidebar.markdown("## MP1 Control Centre")
uploaded=st.sidebar.file_uploader("Upload Excel or CSV",type=["xlsx","xlsm","csv"])
profile_block()

if uploaded is None:
    c1,c2,c3=st.columns(3)
    with c1: st.markdown("<div class='mp-card'><h3>① Upload</h3>Use your own Excel or CSV dataset. Multiple worksheets are supported.</div>",unsafe_allow_html=True)
    with c2: st.markdown("<div class='mp-card'><h3>② Model</h3>Select SLR/MLR for continuous Y or logistic regression for binary Y.</div>",unsafe_allow_html=True)
    with c3: st.markdown("<div class='mp-card'><h3>③ Interpret</h3>Review inference, assumptions, visual diagnostics and export the analysis.</div>",unsafe_allow_html=True)
    st.info("Upload a dataset from the left sidebar to begin.")
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

st.markdown("### Dataset overview")
m1,m2,m3,m4=st.columns(4); m1.metric("Rows",f"{len(df):,}"); m2.metric("Columns",len(df.columns)); m3.metric("Numeric candidates",len(nums)); m4.metric("Missing cells",f"{int(df.isna().sum().sum()):,}")
with st.expander("Preview uploaded data",expanded=False): st.dataframe(df.head(100),use_container_width=True,height=330)

if not run:
    st.markdown("<div class='callout'><b>Ready:</b> choose the model and variables in the MP1 Control Centre, then click <b>Run Regression Analysis</b>.</div>",unsafe_allow_html=True)
    st.stop()
if not x_cols: st.error("Select at least one X variable."); st.stop()

try:
    if model_choice.startswith("Linear"):
        work,dropped=prepare_linear(df,y_col,x_cols,missing); model,coeff,pred,vif,diag=fit_linear(work,y_col,x_cols,confidence)
        model_name="Simple Linear Regression" if len(x_cols)==1 else "Multiple Linear Regression"
        rmse=math.sqrt(np.mean(model.resid**2)); mae=np.mean(np.abs(model.resid))
        st.markdown(f"## {model_name} results")
        c=st.columns(6); c[0].metric("Observations",int(model.nobs)); c[1].metric("R²",f"{model.rsquared:.4f}"); c[2].metric("Adjusted R²",f"{model.rsquared_adj:.4f}"); c[3].metric("Model p-value",f"{model.f_pvalue:.4g}"); c[4].metric("RMSE",f"{rmse:.4f}"); c[5].metric("MAE",f"{mae:.4f}")
        tabs=st.tabs(["Executive Summary","Coefficients","Visual Analysis","Diagnostics","Predictions","Learning Corner","Excel Report"])
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
        report=excel_report(model_name,uploaded.name,sheet,{"Analysis Data":work,"Coefficients":coeff,"Diagnostics":diag,"VIF":vif,"Predictions":pred},[("Y Variable",y_col),("X Variables",", ".join(x_cols)),("R-squared",model.rsquared),("Adjusted R-squared",model.rsquared_adj),("Model p-value",model.f_pvalue),("RMSE",rmse)])
        with tabs[6]: st.download_button("⬇ Download formatted Excel report",report,f"MP1_{model_name.replace(' ','_')}_{datetime.now():%Y%m%d}.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)
    else:
        work,mapping,dropped=prepare_logistic(df,y_col,x_cols,missing,event_label); model,coeff,pred,vif,metrics,matrix,cal=fit_logistic(work,y_col,x_cols,confidence,cutoff)
        mv=metrics.set_index("Metric")["Value"]
        st.markdown("## Binary Logistic Regression results")
        c=st.columns(6); c[0].metric("Observations",int(mv["Observations"])); c[1].metric("ROC AUC",f"{mv['ROC AUC']:.4f}"); c[2].metric("Accuracy",f"{mv['Accuracy']:.1%}"); c[3].metric("Sensitivity",f"{mv['Sensitivity / Recall']:.1%}"); c[4].metric("Specificity",f"{mv['Specificity']:.1%}"); c[5].metric("F1 Score",f"{mv['F1 Score']:.4f}")
        tabs=st.tabs(["Executive Summary","Odds Ratios","ROC & Classification","Diagnostics","Predictions","Learning Corner","Excel Report"])
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
        report=excel_report("Binary Logistic Regression",uploaded.name,sheet,{"Analysis Data":work,"Class Mapping":pd.DataFrame([{"Original Category":k,"Binary Code":v} for k,v in mapping.items()]),"Model Metrics":metrics,"Odds Ratios":coeff,"Confusion Matrix":matrix,"HL Calibration":cal,"VIF":vif,"Predictions":pred},[("Y Variable",y_col),("Event coded 1",str(event_label)),("X Variables",", ".join(x_cols)),("ROC AUC",mv["ROC AUC"]),("Accuracy",mv["Accuracy"]),("Classification cutoff",cutoff)])
        with tabs[6]: st.download_button("⬇ Download formatted Excel report",report,f"MP1_Logistic_Regression_{datetime.now():%Y%m%d}.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)
except Exception as e:
    st.error(f"Analysis could not be completed: {e}")
    st.info("Check that the selected variables contain usable values, Y is appropriate for the chosen model, both logistic classes are represented, and the number of predictors is reasonable for the sample size.")

st.markdown("""<div class='footer'><b>Prof. V. Ravichandran</b> • Founder, The Mountain Path Academy<br>Applied Finance • Risk Management • Financial Analytics • Python & Excel Learning<br><a href='https://themountainpathacademy.com/' target='_blank'>themountainpathacademy.com</a><br><small>Educational analytics application. Model outputs are estimates and should be interpreted with statistical and domain judgement.</small></div>""",unsafe_allow_html=True)

