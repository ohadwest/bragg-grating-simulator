import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import streamlit as st

# ==========================================
# הגדרות עמוד ראשיות
# ==========================================
MODULE_VERSION = "v1.2.0"

st.set_page_config(
    page_title="Bragg Grating Simulator",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# מתג שפה (He / En)
# ==========================================
top_col1, top_col2 = st.columns([8, 2])
with top_col2:
    lang = st.radio(
        "Language / שפה",
        options=["He", "En"],
        horizontal=True,
        index=0,
        label_visibility="collapsed"
    )

T = {
    "He": {
        "dir": "rtl",
        "align": "right",
        "title": "Bragg Grating & DBR Simulation Engine",
        "subtitle": "סימולטור 1D Transfer Matrix Method למראות פוטוניות וחללי תהודה (Cavities)",
        "tab1": "1. מראת בראג בודדת (DBR)",
        "tab2": "2. מבנה הפרעה / מהוד (Defect Cavity)",
        "tab3": "3. משוואות ופיזיקה",
        "params": "⚙️ פרמטרי קלט",
        "wl_center": "אורך גל לתכנון λ₀ (nm):",
        "n_inc": "מקדם שבירה כניסה (Incident):",
        "n_sub": "מקדם שבירה מצע (Substrate):",
        "layer_config": "🔲 הגדרת השכבות",
        "n1_label": "מקדם שבירה n₁ (גבוה):",
        "n2_label": "מקדם שבירה n₂ (נמוך):",
        "n_pairs": "מספר מחזורים N (זוגות):",
        "dev_label": "סטייה מעובי רבע-גל (%):",
        "dev_help": "מאפשר לשנות את העובי בסטייה של X אחוזים מעובי הרבע-גל האידיאלי (λ/4n)",
        "peak_r": "החזרה מקסימלית (R_peak)",
        "stopband": "רוחב הפס החוסם (Stopband Δλ)",
        "download_lin": "📥 הורד גרף ליניארי (PNG)",
        "download_log": "📥 הורד גרף לוגריתמי / dB (PNG)",
        # Defect Cavity Texts
        "top_mirror": "🔹 מראה עליונה (Top Mirror)",
        "bot_mirror": "🔸 מראה תחתונה (Bottom Mirror)",
        "defect_layer": "❌ שכבת הפרעה (Defect / Cavity)",
        "n_def": "מקדם שבירה של ההפרעה (n_def):",
        "d_def": "עובי ההפרעה בננומטר (d_def):",
        "use_quarter_wave": "השתמש בחצי-גל λ₀/(2n_def)",
        "calc_q": "Q-Factor מחושב:",
        "calc_fwhm": "רוחב חצי-מקסימום (FWHM):",
        "graph_linear_title": "📊 ספקטרום בסקאלה ליניארית (Linear Scale)",
        "graph_log_title": "📈 ספקטרום בסקאלה לוגריתמית (dB Scale)",
        "footer": "Powered by Transfer Matrix Method (TMM) | Engineered for Silicon Photonics"
    },
    "En": {
        "dir": "ltr",
        "align": "left",
        "title": "Bragg Grating & DBR Simulation Engine",
        "subtitle": "1D Transfer Matrix Method (TMM) Simulator for Photonic Mirrors & Cavities",
        "tab1": "1. Single Bragg Mirror (DBR)",
        "tab2": "2. Defect Cavity (FP Filter)",
        "tab3": "3. Equations & Physics",
        "params": "⚙️ Input Parameters",
        "wl_center": "Design Wavelength λ₀ (nm):",
        "n_inc": "Incident Index (n_inc):",
        "n_sub": "Substrate Index (n_sub):",
        "layer_config": "🔲 Layer Configuration",
        "n1_label": "Refractive Index n₁ (High):",
        "n2_label": "Refractive Index n₂ (Low):",
        "n_pairs": "Number of Periods N (Pairs):",
        "dev_label": "Deviation from λ/4 (%):",
        "dev_help": "Deviate the layer thickness by X percent from the ideal quarter-wave condition (λ/4n)",
        "peak_r": "Peak Reflectivity (R_peak)",
        "stopband": "Stopband Width (Δλ)",
        "download_lin": "📥 Download Linear Plot (PNG)",
        "download_log": "📥 Download Logarithmic/dB Plot (PNG)",
        # Defect Cavity Texts
        "top_mirror": "🔹 Top Mirror",
        "bot_mirror": "🔸 Bottom Mirror",
        "defect_layer": "❌ Defect Layer (Cavity)",
        "n_def": "Defect Refractive Index (n_def):",
        "d_def": "Defect Thickness (nm):",
        "use_quarter_wave": "Use half-wave λ₀/(2n_def)",
        "calc_q": "Calculated Q-Factor:",
        "calc_fwhm": "FWHM:",
        "graph_linear_title": "📊 Linear Scale Spectrum",
        "graph_log_title": "📈 Logarithmic / dB Scale Spectrum",
        "footer": "Powered by Transfer Matrix Method (TMM) | Engineered for Silicon Photonics"
    }
}[lang]

# ==========================================
# עיצוב CSS דינמי
# ==========================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Heebo:wght@400;600;800&family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', 'Heebo', sans-serif; direction: {T['dir']}; }}
    
    .main-title {{ font-size: 2.8rem; font-weight: 900; background: linear-gradient(90deg, #F59E0B 0%, #EC4899 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 0px; padding-top: 0.2rem; }}
    .sub-title {{ text-align: center; color: #475569; font-size: 1.15rem; margin-bottom: 30px; font-weight: 700; }}
    .metric-card {{ background-color: #0F172A; border: 1px solid #334155; border-radius: 10px; padding: 15px; text-align: center; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
    .metric-val {{ font-size: 1.8rem; font-weight: 900; color: #38BDF8; }}
    .metric-label {{ font-size: 0.9rem; color: #94A3B8; font-weight: 600; }}
    .stDownloadButton > button {{ background-color: #1E293B !important; color: #38BDF8 !important; border: 1px solid #38BDF8 !important; font-weight: 700 !important; border-radius: 8px !important; width: 100%; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f'<div class="main-title">{T["title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">{T["subtitle"]}</div>', unsafe_allow_html=True)

# ==========================================
# מנוע חישוב פיזיקלי (Transfer Matrix Method)
# ==========================================
def tmm_1d(wls_nm, layers, n_inc, n_sub):
    """ TMM Engine for normal incidence. layers = [(n, d_nm), ...] """
    R_out, T_out = [], []
    for wl in wls_nm:
        M = np.identity(2, dtype=complex)
        for n, d in layers:
            delta = (2.0 * np.pi / wl) * n * d
            m_layer = np.array([
                [np.cos(delta), -1j / n * np.sin(delta)],
                [-1j * n * np.sin(delta), np.cos(delta)]
            ], dtype=complex)
            M = M @ m_layer
        
        m11, m12 = M[0, 0], M[0, 1]
        m21, m22 = M[1, 0], M[1, 1]
        
        r = ((m11 + m12 * n_sub) * n_inc - (m21 + m22 * n_sub)) / ((m11 + m12 * n_sub) * n_inc + (m21 + m22 * n_sub))
        t = (2.0 * n_inc) / ((m11 + m12 * n_sub) * n_inc + (m21 + m22 * n_sub))
        
        R_out.append(np.abs(r)**2)
        T_out.append((np.real(n_sub) / np.real(n_inc)) * np.abs(t)**2)
        
    return np.array(R_out), np.array(T_out)

def figure_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf

# ==========================================
# מבנה לשוניות
# ==========================================
tab1, tab2, tab3 = st.tabs([T["tab1"], T["tab2"], T["tab3"]])

# ==========================================
# TAB 1: Single Bragg Mirror (DBR)
# ==========================================
with tab1:
    col_s1, col_m1 = st.columns([1.2, 2.5], gap="large")
    
    with col_s1:
        st.subheader(T["params"])
        wl0_1 = st.number_input(T["wl_center"], value=1550.0, step=1.0, key="wl1")
        n_inc_1 = st.number_input(T["n_inc"], value=1.0, step=0.01, key="ni1")
        n_sub_1 = st.number_input(T["n_sub"], value=1.45, step=0.01, key="ns1")
        
        st.divider()
        st.markdown(f"#### {T['layer_config']}")
        n1_1 = st.number_input(T["n1_label"], value=2.0, step=0.01, key="n1_1")
        dev1_1 = st.slider(T["dev_label"], -50.0, 50.0, 0.0, step=1.0, key="dev1_1", help=T["dev_help"])
        
        n2_1 = st.number_input(T["n2_label"], value=1.45, step=0.01, key="n2_1")
        dev2_1 = st.slider(T["dev_label"], -50.0, 50.0, 0.0, step=1.0, key="dev2_1", help=T["dev_help"])
        
        N_pairs_1 = st.number_input(T["n_pairs"], value=15, min_value=1, step=1, key="np1")
        
        d1_1 = (wl0_1 / (4 * n1_1)) * (1.0 + dev1_1 / 100.0)
        d2_1 = (wl0_1 / (4 * n2_1)) * (1.0 + dev2_1 / 100.0)
        
        st.info(f"**Applied Thicknesses:**\n* $d_1 = {d1_1:.2f}$ nm\n* $d_2 = {d2_1:.2f}$ nm")

    with col_m1:
        layers_dbr = []
        for _ in range(int(N_pairs_1)):
            layers_dbr.append((n1_1, d1_1))
            layers_dbr.append((n2_1, d2_1))
            
        wls_1 = np.linspace(wl0_1 - 400, wl0_1 + 400, 2000)
        R1, T1 = tmm_1d(wls_1, layers_dbr, n_inc_1, n_sub_1)
        
        stopband_analy = (4 * wl0_1 / np.pi) * np.arcsin(np.abs(n1_1 - n2_1) / (n1_1 + n2_1))
        
        mc1, mc2, mc3 = st.columns(3)
        mc1.markdown(f'<div class="metric-card"><div class="metric-label">{T["peak_r"]}</div><div class="metric-val">{np.max(R1)*100:.2f}%</div></div>', unsafe_allow_html=True)
        sb_display = f"~{stopband_analy:.1f} nm" if (dev1_1==0 and dev2_1==0) else "Deviated"
        mc2.markdown(f'<div class="metric-card"><div class="metric-label">{T["stopband"]}</div><div class="metric-val">{sb_display}</div></div>', unsafe_allow_html=True)
        mc3.markdown(f'<div class="metric-card"><div class="metric-label">Max Transmission</div><div class="metric-val">{np.max(T1)*100:.2f}%</div></div>', unsafe_allow_html=True)
        
        # --- 1. גרף בסקאלה ליניארית ---
        st.markdown(f"#### {T['graph_linear_title']}")
        fig1_lin, ax1_lin = plt.subplots(figsize=(9, 3.8), dpi=150)
        fig1_lin.patch.set_facecolor('#0F172A')
        ax1_lin.set_facecolor('#0F172A')
        
        ax1_lin.plot(wls_1, R1, color='#38BDF8', linewidth=2, label='Reflectivity (R)')
        ax1_lin.plot(wls_1, T1, color='#F43F5E', linewidth=2, linestyle='--', alpha=0.8, label='Transmission (T)')
        ax1_lin.axvline(wl0_1, color='#94A3B8', linestyle=':', label='Design λ₀')
        ax1_lin.set_ylim(-0.02, 1.05)
        ax1_lin.set_xlabel("Wavelength λ (nm)", color='#94A3B8', fontweight='bold')
        ax1_lin.set_ylabel("Linear Value (0-1)", color='#94A3B8', fontweight='bold')
        ax1_lin.grid(True, color='#334155', linestyle=':', alpha=0.6)
        ax1_lin.legend(facecolor='#1E293B', edgecolor='#334155', labelcolor='#F8FAFC')
        
        st.pyplot(fig1_lin)
        st.download_button(T["download_lin"], data=figure_to_bytes(fig1_lin), file_name="dbr_linear.png", mime="image/png", key="dl_lin_1")

        st.write("")
        
        # --- 2. גרף בסקאלה לוגריתמית / dB ---
        st.markdown(f"#### {T['graph_log_title']}")
        fig1_log, ax1_log = plt.subplots(figsize=(9, 3.8), dpi=150)
        fig1_log.patch.set_facecolor('#0F172A')
        ax1_log.set_facecolor('#0F172A')
        
        R1_db = 10 * np.log10(np.maximum(1e-6, R1))
        T1_db = 10 * np.log10(np.maximum(1e-6, T1))
        
        ax1_log.plot(wls_1, T1_db, color='#F43F5E', linewidth=2, label='Transmission (dB)')
        ax1_log.plot(wls_1, R1_db, color='#38BDF8', linewidth=1.5, linestyle='--', alpha=0.7, label='Reflectivity (dB)')
        ax1_log.axvline(wl0_1, color='#94A3B8', linestyle=':', label='Design λ₀')
        ax1_log.set_ylim(-60, 2)
        ax1_log.set_xlabel("Wavelength λ (nm)", color='#94A3B8', fontweight='bold')
        ax1_log.set_ylabel("Power (dB)", color='#94A3B8', fontweight='bold')
        ax1_log.grid(True, color='#334155', linestyle=':', alpha=0.6)
        ax1_log.legend(facecolor='#1E293B', edgecolor='#334155', labelcolor='#F8FAFC')
        
        st.pyplot(fig1_log)
        st.download_button(T["download_log"], data=figure_to_bytes(fig1_log), file_name="dbr_log_db.png", mime="image/png", key="dl_log_1")

# ==========================================
# TAB 2: Defect Cavity (Fabry-Perot)
# ==========================================
with tab2:
    col_s2, col_m2 = st.columns([1.2, 2.5], gap="large")
    
    with col_s2:
        wl0_2 = st.number_input(T["wl_center"], value=1550.0, step=1.0, key="wl2")
        
        st.markdown(f"#### {T['bot_mirror']} & {T['top_mirror']}")
        n_high = st.number_input("n_High (e.g., SiN):", value=2.0, step=0.01)
        n_low = st.number_input("n_Low (e.g., SiO2):", value=1.45, step=0.01)
        N_pairs = st.number_input("Number of Pairs (Top & Bot):", value=10, step=1)
        
        st.markdown(f"#### {T['defect_layer']}")
        n_def = st.number_input(T["n_def"], value=2.0, step=0.01)
        use_half = st.checkbox(T["use_quarter_wave"], value=True, help="Use strictly λ₀/(2n) for center resonance")
        
        if use_half:
            d_def = wl0_2 / (2 * n_def)
            st.info(f"Defect Thickness: {d_def:.2f} nm (λ₀/2)")
        else:
            d_def = st.number_input(T["d_def"], value=387.5, step=1.0)
            
    with col_m2:
        layers_cavity = []
        
        # בניית המראה העליונה (סימטרית: תמיד מתחילה ומסתיימת ב-High)
        d_high = wl0_2 / (4 * n_high)
        d_low = wl0_2 / (4 * n_low)
        
        # מראה עליונה (מתחיל ב-High, מסיים ב-High לפני החלל)
        for _ in range(int(N_pairs)):
            layers_cavity.append((n_high, d_high))
            layers_cavity.append((n_low, d_low))
        layers_cavity.append((n_high, d_high)) # שכבת High מסימת כדי לקבע פאזת החזרה 0 או pi

        # שכבת ההפרעה (Cavity) - עובי חצי-גל ימרכז אותה בדיוק
        layers_cavity.append((n_def, d_def))
        
        # מראה תחתונה (סימטרית: מתחילה ב-High, מסיימת ב-High)
        layers_cavity.append((n_high, d_high))
        for _ in range(int(N_pairs)):
            layers_cavity.append((n_low, d_low))
            layers_cavity.append((n_high, d_high))
            
        # סריקה ברזולוציה גבוהה כדי לא לפספס את הפיק
        wls_2 = np.linspace(wl0_2 - 100, wl0_2 + 100, 10000) 
        R2, T2 = tmm_1d(wls_2, layers_cavity, 1.0, 1.45)
        
        # חילוץ Q-Factor
        peaks, props = find_peaks(T2, prominence=0.05)
        Q_val, fwhm_nm = 0, 0
        if len(peaks) > 0:
            center_peak_idx = peaks[np.argmin(np.abs(wls_2[peaks] - wl0_2))]
            pk_wl = wls_2[center_peak_idx]
            
            half_max = T2[center_peak_idx] / 2.0
            
            left_points = np.where(T2[:center_peak_idx] <= half_max)[0]
            right_points = np.where(T2[center_peak_idx:] <= half_max)[0]
            
            if len(left_points) > 0 and len(right_points) > 0:
                idx_left = left_points[-1]
                idx_right = center_peak_idx + right_points[0]
                fwhm_nm = wls_2[idx_right] - wls_2[idx_left]
                Q_val = pk_wl / fwhm_nm if fwhm_nm > 0 else 0
            
        mc4, mc5 = st.columns(2)
        mc4.markdown(f'<div class="metric-card"><div class="metric-label">{T["calc_q"]}</div><div class="metric-val">{Q_val:,.0f}</div></div>', unsafe_allow_html=True)
        mc5.markdown(f'<div class="metric-card"><div class="metric-label">{T["calc_fwhm"]}</div><div class="metric-val">{fwhm_nm:.3f} nm</div></div>', unsafe_allow_html=True)

        # גרף לוגריתמי / dB 
        st.markdown(f"#### {T['graph_log_title']}")
        fig2_log, ax2_log = plt.subplots(figsize=(9, 4.5), dpi=150)
        fig2_log.patch.set_facecolor('#0F172A')
        ax2_log.set_facecolor('#0F172A')
        
        R2_db = 10 * np.log10(np.maximum(1e-6, R2))
        T2_db = 10 * np.log10(np.maximum(1e-6, T2))
        
        ax2_log.plot(wls_2, T2_db, color='#10B981', linewidth=2, label='Transmission (dB)')
        ax2_log.plot(wls_2, R2_db, color='#38BDF8', linewidth=1.5, linestyle='--', alpha=0.7, label='Reflectivity Notch (dB)')
        
        # קו סימון אמצע - נוכיח שה-Notch יושב בדיוק באמצע!
        ax2_log.axvline(wl0_2, color='#F43F5E', linestyle=':', label='Center $\lambda_0$')
        
        ax2_log.set_ylim(-60, 2)
        ax2_log.set_xlabel("Wavelength λ (nm)", color='#94A3B8', fontweight='bold')
        ax2_log.set_ylabel("Power (dB)", color='#94A3B8', fontweight='bold')
        ax2_log.grid(True, color='#334155', linestyle=':', alpha=0.6)
        ax2_log.legend(facecolor='#1E293B', edgecolor='#334155', labelcolor='#F8FAFC')
        
        st.pyplot(fig2_log)
# ==========================================
# TAB 3: Physics & Equations
# ==========================================
with tab3:
    st.markdown("""
    ### 📐 Transfer Matrix Method (TMM) Equations
    
    The **Transfer Matrix Method** is used to calculate the reflection and transmission spectra of 1D layered structures. For normal incidence, the characteristic matrix $M_i$ of the $i$-th layer with thickness $d_i$ and refractive index $n_i$ is given by:
    
    $$ M_i = \\begin{pmatrix} \\cos\\delta_i & -\\frac{i}{n_i} \\sin\\delta_i \\\\ -i n_i \\sin\\delta_i & \\cos\\delta_i \\end{pmatrix} $$
    
    where the optical phase thickness $\\delta_i$ is:
    
    $$ \\delta_i = \\frac{2\\pi}{\\lambda} n_i d_i $$
    
    The total system matrix $M$ for a multi-layer stack is simply the product of the individual layer matrices:
    
    $$ M = M_1 \\cdot M_2 \\cdots M_N = \\begin{pmatrix} m_{11} & m_{12} \\\\ m_{21} & m_{22} \\end{pmatrix} $$
    
    ### ⚡ Distributed Bragg Reflector (DBR) Properties
    
    For an ideal Bragg mirror consisting of alternating high ($n_1$) and low ($n_2$) index quarter-wave layers ($d_i = \\lambda_0 / 4n_i$), the maximum theoretical reflectivity for $N$ pairs on a substrate $n_s$ is:
    
    $$ R_{\\text{peak}} = \\left( \\frac{n_{\\text{inc}} - n_s (n_1/n_2)^{2N}}{n_{\\text{inc}} + n_s (n_1/n_2)^{2N}} \\right)^2 $$
    
    The width of the high-reflection band (Stopband bandwidth) is approximated by:
    
    $$ \\Delta\\lambda_{\\text{stopband}} \\approx \\frac{4\\lambda_0}{\\pi} \\arcsin\\left( \\frac{|n_1 - n_2|}{n_1 + n_2} \\right) $$
    """)

st.divider()
st.markdown(f'<div style="text-align: center; color: #64748B; font-size: 0.9rem;">{T["footer"]}</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #475569; font-size: 0.85rem; font-weight: 700;">© 2026 ohadwest. All rights reserved.</div>', unsafe_allow_html=True)
