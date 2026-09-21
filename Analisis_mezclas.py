import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import f_oneway

# ---------------------------------------------------------
# 1. CONFIGURACIÓN E INTERFAZ
# ---------------------------------------------------------
st.set_page_config(page_title="DURAPAV - Análisis Factorial", layout="wide")

st.sidebar.markdown("### Laboratorio de Caminos - UPV")
st.sidebar.markdown("**Proyecto DURAPAV**")
st.sidebar.markdown("---")

st.title("Análisis Dinámico de Parámetros: Ensayo Ideal-CT")

# --- DESCRIPCIÓN INICIAL ---
st.markdown("El dashboard permite explorar gráficamente los resultados obtenidos mediante el ensayo Ideal-CT. Su objetivo es facilitar la comparación entre mezclas y condiciones de ensayo, evaluando cómo cambian la tenacidad, la ductilidad, la fragilidad y la resistencia en función del envejecimiento, la incorporación de material fresado (RAP) y la velocidad de ensayo.")

col_desc1, col_desc2 = st.columns(2)
with col_desc1:
    st.info("""
    **Variables empleadas como filtros**
    * **Envejecimiento:** Sin_Envejecer (E0); Nivel_1 (ej. 2 días a 85 ºC); Nivel_2 (ej. 5 días a 85 ºC).
    * **RAP:** Sin_RAP = mezcla sin asfalto recuperado; 30_RAP = mezcla con 30% de asfalto recuperado.
    * **Velocidad:** velocidad de desplazamiento del ensayo, con valores de 1 y 50 mm/min.
    * **Mezcla:** AC16, AC22, BBTM11, SMA16.
    """)
with col_desc2:
    st.info("""
    **Parámetros del ensayo Ideal-CT**
    * **Gf:** tenacidad o energía de fractura (J/m²).
    * **Gf prepico y postpico:** contribución energética antes y después de la carga máxima.
    * **l75:** ductilidad, asociada a la capacidad de deformación.
    * **lpeak:** deformación elástica hasta la carga máxima.
    * **m75 / m_xx:** fragilidad, asociada a la pendiente postpico y propagación de fisura.
    * **Carga Pico:** resistencia asociada al inicio de la formación de la fisura.
    * **CT_Index:** índice global de tolerancia a la fisuración.
    * **Ratio_Flexibilidad:** l75 / m75.
    """)

@st.cache_data
def cargar_datos():
    df = pd.read_excel('datos.xlsx')
    df.columns = df.columns.astype(str).str.strip()
    
    # --- LIMPIEZA A PRUEBA DE FALLOS ---
    df = df.replace('-', np.nan)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    # Forzamos las variables a numéricas por si el Excel tiene guiones de texto
    cols_numericas = ['Carga_pico', 'CT_Index', 'Gf', 'Gf_prepico', 'Gf_postpico', 'm75', 'l75', 'lpeak', 'Rigidez_20']
    for col in cols_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'Espesor' in df.columns and 'Diametro' in df.columns:
        # Tensión en N/mm² (MPa)
        df['Tension_Rotura'] = (df['Carga_pico'] * 1000) / (df['Espesor'] * df['Diametro'])
        
    if 'Mezcla' in df.columns:
        if 'RAP' in df.columns and 'Envejecimiento' in df.columns:
            df['Todas'] = df['Mezcla'] + " | RAP: " + df['RAP'].astype(str) + " | Env: " + df['Envejecimiento'].astype(str)
        else:
            df['Todas'] = df['Mezcla']
    else:
        df['Todas'] = df.iloc[:, 0]
            
    return df

df = cargar_datos()

# Calculamos parámetros globales derivados
if 'l75' in df.columns and 'm75' in df.columns:
    df['abs_m75'] = df['m75'].abs()
    df['Ratio_Flexibilidad'] = (df['l75'] / df['abs_m75'].replace(0, np.nan)) * 1e6

# --- NUEVOS PARÁMETROS EXPERIMENTALES (SISTEMA INTERNACIONAL) ---
if 'Gf_prepico' in df.columns and 'Tension_Rotura' in df.columns and 'abs_m75' in df.columns:
    df['Tension_Rotura_SI'] = df['Tension_Rotura'] * 1e6
    
    # Variante 1: Gf_prepico 
    df['Num_Prepico'] = df['Gf_prepico'] * df['Tension_Rotura_SI']
    df['Indice_Prepico'] = df['Num_Prepico'] / df['abs_m75'].replace(0, np.nan)
    
    # Variante 2: Gf_postpico 
    if 'Gf_postpico' in df.columns:
        df['Num_Postpico'] = df['Gf_postpico'] * df['Tension_Rotura_SI']
        df['Indice_Postpico'] = df['Num_Postpico'] / df['abs_m75'].replace(0, np.nan)
        
    df['TR_sobre_m75'] = df['Tension_Rotura_SI'] / df['abs_m75'].replace(0, np.nan)

st.markdown("---")

# ---------------------------------------------------------
# 2. PESTAÑAS DE NAVEGACIÓN
# ---------------------------------------------------------
tab_parametrico, tab_reg, tab_energia, tab_stiff, tab_fisuracion, tab_disc = st.tabs([
    "Análisis Paramétrico", 
    "Comparativa Rigidez",
    "Reparto Energético",
    "Diagrama Stiff",
    "Índice de fisuración",
    "Poder Discriminatorio"
])

# --- PESTAÑA 1: ANÁLISIS PARAMÉTRICO ---
with tab_parametrico:
    st.header("1. Análisis Paramétrico")
    
    opciones_var = ['Carga_pico', 'CT_Index', 'Gf', 'Gf_prepico', 'Gf_postpico', 'm75', 'l75', 'Ratio_Flexibilidad', 'lpeak']
    if 'Tension_Rotura' in df.columns: opciones_var.append('Tension_Rotura')
    
    st.markdown("### 1.1. Comparativa Directa por Parámetro (Medias)")
    col1A, col1B, col1C = st.columns(3)
    with col1A:
        param_bar = st.selectbox("Parámetro (Eje Y):", opciones_var, key="bar_y")
    with col1B:
        eje_x_bar = st.selectbox("Agrupar por (Eje X):", ['Todas', 'Mezcla', 'Envejecimiento', 'RAP'], key="bar_x")
    with col1C:
        color_bar = st.selectbox("Separar colores por:", ['Velocidad', 'Envejecimiento', 'RAP', 'Mezcla', 'Todas'], key="bar_color")

    agrupacion = [eje_x_bar] if eje_x_bar == color_bar else [eje_x_bar, color_bar]
    df_barras = df.groupby(agrupacion)[param_bar].mean().reset_index()
    df_barras[color_bar] = df_barras[color_bar].astype(str)
    
    modo_barra_t1 = 'relative' if eje_x_bar == color_bar else 'group'
    fmt_texto_t2 = '.4f' if param_bar == 'lpeak' else '.2f'
    
    fig_bar = px.bar(
        df_barras, x=eje_x_bar, y=param_bar, color=color_bar, 
        barmode=modo_barra_t1, text_auto=fmt_texto_t2, color_discrete_sequence=px.colors.qualitative.Alphabet if color_bar == 'Todas' else px.colors.qualitative.Set2
    )
    fig_bar.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    
    st.markdown("### 1.2. Análisis de Dispersión y Variabilidad (Cajas)")
    st.markdown("Este gráfico muestra todos los resultados individuales. Las cajas representan dónde se agrupa el 50% de las probetas, la línea central es la mediana, y los puntos sueltos son valores atípicos (*outliers*).")
    
    col1D, col1E, col1F, col1G = st.columns(4)
    with col1D:
        mezclas_disp_t3 = df['Mezcla'].dropna().unique().tolist()
        filtro_mezcla_t3 = st.selectbox("Filtrar Mezcla:", ["Todas"] + mezclas_disp_t3, key="box_mezcla")
    with col1E:
        param_box = st.selectbox("Parámetro (Eje Y):", opciones_var, key="box_y")
    with col1F:
        eje_x_box = st.selectbox("Agrupar por (Eje X):", ['Todas', 'Mezcla', 'Envejecimiento', 'RAP'], key="box_x")
    with col1G:
        color_box = st.selectbox("Separar colores por:", ['Velocidad', 'RAP', 'Envejecimiento', 'Mezcla', 'Todas'], key="box_color")

    if filtro_mezcla_t3 != "Todas":
        df_cajas = df[df['Mezcla'] == filtro_mezcla_t3].copy()
    else:
        df_cajas = df.copy()

    df_cajas[color_box] = df_cajas[color_box].astype(str)

    fig_box = px.box(
        df_cajas, x=eje_x_box, y=param_box, color=color_box, 
        points="all", 
        color_discrete_sequence=px.colors.qualitative.Alphabet if color_box == 'Todas' else px.colors.qualitative.Pastel
    )
    fig_box.update_layout(height=600, xaxis_tickangle=-45)
    st.plotly_chart(fig_box, use_container_width=True)


# --- PESTAÑA 2: COMPARATIVA RIGIDEZ ---
with tab_reg:
    st.header("2. Comparativa Rigidez")
    st.markdown("⚠️ *Este análisis de regresión excluye los ensayos a bajas velocidades, mostrando únicamente los resultados a **50 mm/min**.*")
    
    df_tab4 = df[df['Velocidad'] == 50].copy()
    
    mezclas_unicas = df_tab4['Mezcla'].dropna().unique().tolist()
    mezcla_elegida = st.selectbox(
        "Filtra los datos del modelo estadístico:", 
        ['Todas'] + mezclas_unicas,
        help="Si eliges 'Todas', el modelo de regresión evaluará el comportamiento global."
    )
    
    if mezcla_elegida == 'Todas':
        df_est = df_tab4.copy()
    else:
        df_est = df_tab4[df_tab4['Mezcla'] == mezcla_elegida].copy()
        
    st.markdown("---")
    
    st.subheader("2.1. Influencia de la Pendiente Pre-Pico")
    lista_pendientes = [
        'm30_10', 'm35_10', 'm40_10', 'm45_10', 'm50_10', 'm55_10', 'm60_10', 'm65_10', 'm70_10', 
        'm30_20', 'm35_20', 'm40_20', 'm45_20', 'm50_20', 'm55_20', 'm60_20', 'm65_20', 'm70_20'
    ]
    pendientes_validas = [p for p in lista_pendientes if p in df_est.columns]
    
    col_2A, col_2B = st.columns([1, 2])
    with col_2A:
        pend_selec = st.selectbox("Selecciona la pendiente a evaluar:", pendientes_validas)
        df_clean = df_est.dropna(subset=[pend_selec, 'Rigidez_20'])
        
        if len(df_clean) > 2:
            X_simple = sm.add_constant(df_clean[pend_selec])
            modelo_simple = sm.OLS(df_clean['Rigidez_20'], X_simple).fit()
            st.metric(label="Precisión del ajuste (R²)", value=f"{modelo_simple.rsquared:.4f}")
            p_val = modelo_simple.pvalues[pend_selec]
            if p_val < 0.05:
                st.success(f"**Significativo** (p-value: {p_val:.4f})")
            else:
                st.warning(f"**No significativo** (p-value: {p_val:.4f})")
        else:
            st.warning("Faltan datos para realizar la regresión.")
            
    with col_2B:
        if len(df_clean) > 2:
            fig_p, ax_p = plt.subplots(figsize=(9, 5))
            sns.scatterplot(data=df_clean, x=pend_selec, y='Rigidez_20', hue='Todas', palette='tab10', s=90, alpha=0.8, ax=ax_p)
            sns.regplot(data=df_clean, x=pend_selec, y='Rigidez_20', scatter=False, color='black', line_kws={'linestyle': '--', 'alpha':0.6}, ax=ax_p)
            if 'Probeta' in df_clean.columns:
                for idx, row in df_clean.iterrows():
                    if pd.notna(row['Probeta']):
                        ax_p.annotate(str(row['Probeta']), (row[pend_selec], row['Rigidez_20']), textcoords="offset points", xytext=(6, 6), ha='left', fontsize=8, alpha=0.8)
            ax_p.set_ylabel('Rigidez Real (MPa)')
            ax_p.grid(True, linestyle='--', alpha=0.5)
            ax_p.legend(title='Tipo de Probeta', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title_fontsize=9)
            fig_p.tight_layout()
            st.pyplot(fig_p)

    st.markdown("---")
    
    st.subheader("2.2. Parámetros de Rotura vs Rigidez")
    col_2C, col_2D = st.columns(2)
    with col_2C:
        st.markdown("**Carga Pico**")
        df_carga = df_est.dropna(subset=['Carga_pico', 'Rigidez_20'])
        if len(df_carga) > 2:
            X_carga = sm.add_constant(df_carga['Carga_pico'])
            mod_c = sm.OLS(df_carga['Rigidez_20'], X_carga).fit()
            fig_c, ax_c = plt.subplots(figsize=(7, 5))
            sns.scatterplot(data=df_carga, x='Carga_pico', y='Rigidez_20', hue='Todas', palette='tab10', s=80, alpha=0.8, ax=ax_c)
            sns.regplot(data=df_carga, x='Carga_pico', y='Rigidez_20', scatter=False, color='black', line_kws={'linestyle': '--', 'alpha':0.6}, ax=ax_c)
            ax_c.set_xlabel('Carga Pico (N/kN)')
            ax_c.set_ylabel('Rigidez (MPa)')
            ax_c.grid(True, alpha=0.3)
            ax_c.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
            fig_c.tight_layout()
            st.pyplot(fig_c)
            st.caption(f"R²: {mod_c.rsquared:.4f} | p-value: {mod_c.pvalues['Carga_pico']:.4f}")
            
    with col_2D:
        st.markdown("**Tensión de Rotura**")
        if 'Tension_Rotura' in df_est.columns:
            df_tens = df_est.dropna(subset=['Tension_Rotura', 'Rigidez_20'])
            if len(df_tens) > 2:
                X_tens = sm.add_constant(df_tens['Tension_Rotura'])
                mod_t = sm.OLS(df_tens['Rigidez_20'], X_tens).fit()
                fig_t, ax_t = plt.subplots(figsize=(7, 5))
                sns.scatterplot(data=df_tens, x='Tension_Rotura', y='Rigidez_20', hue='Todas', palette='tab10', s=80, alpha=0.8, ax=ax_t)
                sns.regplot(data=df_tens, x='Tension_Rotura', y='Rigidez_20', scatter=False, color='black', line_kws={'linestyle': '--', 'alpha':0.6}, ax=ax_t)
                ax_t.set_xlabel('Tensión Rotura (MPa)')
                ax_t.set_ylabel('Rigidez (MPa)')
                ax_t.grid(True, alpha=0.3)
                ax_t.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
                fig_t.tight_layout()
                st.pyplot(fig_t)
                st.caption(f"R²: {mod_t.rsquared:.4f} | p-value: {mod_t.pvalues['Tension_Rotura']:.4f}")

    st.markdown("---")
    st.subheader("2.3. Matrices de Correlación")
    macro_params = ['Rigidez_20', 'Carga_pico']
    if 'Tension_Rotura' in df_est.columns:
        macro_params.append('Tension_Rotura')
        
    if len(df_est) > 2 and pendientes_validas:
        st.markdown("**A. Correlación cruzada: Parámetros Macro (Rigidez y Rotura) vs Pendientes pre-pico**")
        corr_full = df_est[macro_params + pendientes_validas].corr()
        corr_macro_pendientes = corr_full.loc[macro_params, macro_params + pendientes_validas]
        fig_heat1, ax_heat1 = plt.subplots(figsize=(20, len(macro_params) * 1.5))
        sns.heatmap(corr_macro_pendientes, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=0.5, ax=ax_heat1, annot_kws={"size": 9})
        ax_heat1.tick_params(axis='y', rotation=0)
        st.pyplot(fig_heat1)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**B. Correlación interna: Análisis de colinealidad entre Pendientes**")
        corr_pendientes = df_est[pendientes_validas].corr()
        mask = np.triu(np.ones_like(corr_pendientes, dtype=bool))
        fig_heat2, ax_heat2 = plt.subplots(figsize=(16, 12))
        sns.heatmap(corr_pendientes, mask=mask, annot=True, cmap='YlGnBu', fmt=".4f", linewidths=0.5, ax=ax_heat2, annot_kws={"size": 8})
        st.pyplot(fig_heat2)


# --- PESTAÑA 3: REPARTO ENERGÉTICO ---
with tab_energia:
    st.header("3. Balance Energético de Fractura")
    if 'Gf_prepico' in df.columns and 'Gf_postpico' in df.columns:
        col3A, col3B, col3C = st.columns(3)
        with col3A:
            mezclas_disp = df['Mezcla'].dropna().unique().tolist()
            filtro_mezcla_t5 = st.selectbox("Elegir Mezcla:", ["Todas"] + mezclas_disp, key="t5_mezcla")
        with col3B:
            eje_x_energia = st.selectbox("Agrupar gráficas por (Eje X):", ["Todas", "Envejecimiento", "RAP", "Velocidad"], key="t5_var")
        with col3C:
            tipo_vista = st.radio("Modo de representación:", ["Valores Absolutos (J/m²)", "Porcentaje Relativo (100%)"], horizontal=False, key="energia_modo")
            
        st.markdown("---")
        
        if filtro_mezcla_t5 != "Todas":
            df_energia = df[df['Mezcla'] == filtro_mezcla_t5].copy()
        else:
            df_energia = df.copy()
            
        df_energia = df_energia[~((df_energia['Mezcla'] == 'AC22') & (df_energia['RAP'] != 'Sin_RAP'))]
            
        if not df_energia.empty:
            col_v50, col_v2 = st.columns(2)
            for vel, col_layout in zip([50, 2], [col_v50, col_v2]):
                with col_layout:
                    st.subheader(f"Velocidad: {vel} mm/min")
                    df_vel = df_energia[df_energia['Velocidad'] == vel]
                    
                    if not df_vel.empty:
                        df_g_mean = df_vel.groupby(eje_x_energia)[['Gf_prepico', 'Gf_postpico']].mean().reset_index()
                        if tipo_vista == "Porcentaje Relativo (100%)":
                            total_gf = df_g_mean['Gf_prepico'] + df_g_mean['Gf_postpico']
                            df_g_mean['Gf_prepico'] = (df_g_mean['Gf_prepico'] / total_gf) * 100
                            df_g_mean['Gf_postpico'] = (df_g_mean['Gf_postpico'] / total_gf) * 100
                            eje_y_titulo = "Porcentaje de Energía (%)"
                        else:
                            eje_y_titulo = "Energía (J/m²)"
                            
                        df_melted = pd.melt(df_g_mean, id_vars=[eje_x_energia], value_vars=['Gf_prepico', 'Gf_postpico'], var_name='Fase_Energia', value_name='Valor')
                        fig_stack = px.bar(df_melted, x=eje_x_energia, y='Valor', color='Fase_Energia', barmode='stack', text_auto='.1f', color_discrete_map={'Gf_prepico': '#3498db', 'Gf_postpico': '#e67e22'})
                        fig_stack.update_layout(height=450, xaxis_tickangle=-45, yaxis_title=eje_y_titulo)
                        st.plotly_chart(fig_stack, use_container_width=True)
                    else:
                        st.info(f"No hay datos para {vel} mm/min con los filtros actuales.")
        else:
            st.warning("No hay datos disponibles tras aplicar los filtros de exclusión.")


# --- PESTAÑA 4: DIAGRAMA STIFF ---
with tab_stiff:
    st.header("4. Firma Mecánica: Diagrama de Stiff Diferencial")
    
    cols_stiff = ['Carga_pico', 'l75', 'm75', 'Gf_postpico', 'Rigidez_20', 'Ratio_Flexibilidad']
    missing_cols = [c for c in cols_stiff if c not in df.columns]
    
    if not missing_cols:
        df_50_stiff = df[df['Velocidad'] == 50].copy()
        
        col_setupA, col_setupB = st.columns(2)
        mezclas_stiff = df_50_stiff['Mezcla'].dropna().unique().tolist()
        rap_stiff = df_50_stiff['RAP'].dropna().unique().tolist()
        env_stiff = df_50_stiff['Envejecimiento'].dropna().unique().tolist()

        with col_setupA:
            st.markdown("🟦 **Estado de Referencia**")
            cA1, cA2, cA3 = st.columns(3)
            m_A = cA1.selectbox("Mezcla", mezclas_stiff, key="m_A")
            r_A = cA2.selectbox("RAP", rap_stiff, key="r_A")
            e_A = cA3.selectbox("Env", env_stiff, key="e_A")

        with col_setupB:
            st.markdown("🟧 **Estado Modificado**")
            cB1, cB2, cB3 = st.columns(3)
            m_B = cB1.selectbox("Mezcla", mezclas_stiff, key="m_B")
            r_B = cB2.selectbox("RAP", rap_stiff, key="r_B")
            e_B = cB3.selectbox("Env", env_stiff, key="e_B")
            
        df_A = df_50_stiff[(df_50_stiff['Mezcla'] == m_A) & (df_50_stiff['RAP'] == r_A) & (df_50_stiff['Envejecimiento'] == e_A)]
        df_B = df_50_stiff[(df_50_stiff['Mezcla'] == m_B) & (df_50_stiff['RAP'] == r_B) & (df_50_stiff['Envejecimiento'] == e_B)]
            
        if not df_A.empty and not df_B.empty:
            def norm_stiff(val, col):
                v_min = df_50_stiff[col].min()
                v_max = df_50_stiff[col].max()
                if v_max == v_min: return 0.5
                return (val - v_min) / (v_max - v_min)

            def get_stiff_data(df_subset):
                vals = {
                    'c': df_subset['Carga_pico'].mean(),
                    'l': df_subset['l75'].mean(),
                    'm': df_subset['abs_m75'].mean(),
                    'gpo': df_subset['Gf_postpico'].mean(),
                    'r': df_subset['Rigidez_20'].mean(),
                    'rf': df_subset['Ratio_Flexibilidad'].mean()
                }
                
                x_norm = [
                    -norm_stiff(vals['c'], 'Carga_pico'), 
                    -norm_stiff(vals['m'], 'abs_m75'), 
                    -norm_stiff(vals['r'], 'Rigidez_20'), 
                    norm_stiff(vals['rf'], 'Ratio_Flexibilidad'), 
                    norm_stiff(vals['gpo'], 'Gf_postpico'), 
                    norm_stiff(vals['l'], 'l75')
                ]
                text_vals = [
                    f"{vals['c']:.1f}", f"{vals['m']:.2f}", f"{vals['r']:.1f}", 
                    f"{vals['rf']:.3f}", f"{vals['gpo']:.0f}", f"{vals['l']:.2f}"
                ]
                y_coords = [3, 2, 1, 1, 2, 3]
                x_centroid = sum(x_norm) / 6
                
                x_norm.append(x_norm[0])
                y_coords.append(y_coords[0])
                text_vals.append(text_vals[0])
                return x_norm, y_coords, text_vals, x_centroid

            xA, yA, txtA, cxA = get_stiff_data(df_A)
            xB, yB, txtB, cxB = get_stiff_data(df_B)
            
            fig_stiff = go.Figure()
            
            fig_stiff.add_trace(go.Scatter(
                x=xA, y=yA, fill='toself', fillcolor='rgba(52, 152, 219, 0.4)',
                line=dict(color='#2980b9', width=2), mode='lines+markers+text',
                text=txtA, textposition=["middle right", "middle right", "middle right", "middle left", "middle left", "middle left", "middle right"],
                marker=dict(size=8, color='#2980b9'), textfont=dict(color='#2980b9', size=11), name=f"Ref: {m_A}|{r_A}|{e_A}"
            ))
            
            fig_stiff.add_trace(go.Scatter(
                x=xB, y=yB, fill='toself', fillcolor='rgba(230, 126, 34, 0.4)',
                line=dict(color='#d35400', width=2), mode='lines+markers+text',
                text=txtB, textposition=["middle left", "middle left", "middle left", "middle right", "middle right", "middle right", "middle left"],
                marker=dict(size=8, color='#d35400'), textfont=dict(color='#d35400', size=11), name=f"Mod: {m_B}|{r_B}|{e_B}"
            ))
            
            fig_stiff.add_trace(go.Scatter(
                x=[cxA, cxB], y=[2, 2], mode='markers',
                marker=dict(size=12, color=['#2980b9', '#d35400'], symbol='diamond'),
                name="Centroides"
            ))
            
            fig_stiff.add_shape(type="line", x0=0, y0=0.5, x1=0, y1=3.5, line=dict(color="black", width=2, dash="dash"))
            fig_stiff.add_shape(type="line", x0=-1.1, y0=3, x1=1.1, y1=3, line=dict(color="gray", width=1, dash="dot"))
            fig_stiff.add_shape(type="line", x0=-1.1, y0=2, x1=1.1, y1=2, line=dict(color="gray", width=1, dash="dot"))
            fig_stiff.add_shape(type="line", x0=-1.1, y0=1, x1=1.1, y1=1, line=dict(color="gray", width=1, dash="dot"))
            
            delta_x = cxB - cxA
            tendencia = "Rigidización" if delta_x < 0 else "Ductilización"
            
            fig_stiff.add_annotation(
                x=cxB, y=2, ax=cxA, ay=2, xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=2, arrowcolor="black"
            )
            fig_stiff.add_annotation(
                x=(cxA + cxB)/2, y=2.15, xref="x", yref="y",
                text=f"<b>Migración: {delta_x:.3f} ({tendencia})</b>",
                showarrow=False, font=dict(size=13, color="black"), bgcolor="rgba(255,255,255,0.8)"
            )

            anotaciones_eje = [
                dict(x=-1.2, y=3, text="Carga Pico", xanchor='right', showarrow=False, font=dict(size=12, color="black")),
                dict(x=1.2, y=3, text="l75 (Desplazam.)", xanchor='left', showarrow=False, font=dict(size=12, color="black")),
                dict(x=-1.2, y=2, text="|m75| (Fragilidad)", xanchor='right', showarrow=False, font=dict(size=12, color="black")),
                dict(x=1.2, y=2, text="Gf Post-pico", xanchor='left', showarrow=False, font=dict(size=12, color="black")),
                dict(x=-1.2, y=1, text="Rigidez_20", xanchor='right', showarrow=False, font=dict(size=12, color="black")),
                dict(x=1.2, y=1, text="Ratio Flexibilidad", xanchor='left', showarrow=False, font=dict(size=12, color="black"))
            ]
            
            fig_stiff.update_layout(
                xaxis=dict(
                    range=[-1.5, 1.5], 
                    showticklabels=True, 
                    tickvals=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],
                    ticktext=['1.0', '0.75', '0.5', '0.25', '0', '0.25', '0.5', '0.75', '1.0'],
                    zeroline=False, showgrid=True, gridcolor='rgba(200,200,200,0.2)'
                ),
                yaxis=dict(range=[0.5, 3.5], showticklabels=False, zeroline=False, showgrid=False),
                annotations=list(fig_stiff.layout.annotations) + anotaciones_eje,
                height=650, margin=dict(l=20, r=20, t=40, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_stiff, use_container_width=True)
        else:
            st.warning("No hay datos para alguna de las combinaciones seleccionadas.")
            
        st.markdown("---")
        st.subheader("4.1. Justificación Estadística del Diagrama Stiff")
        st.markdown("Este apartado demuestra matemáticamente la correcta distribución de los ejes (Izquierda = Rigidez/Fragilidad vs Derecha = Ductilidad/Flexibilidad) utilizando los datos a **50 mm/min**.")
        
        val_params = ['Carga_pico', 'l75', 'Gf_prepico', 'Gf_postpico', 'Rigidez_20', 'm75', 'Ratio_Flexibilidad']
        missing_val_cols = [c for c in val_params if c not in df.columns]
        
        if not missing_val_cols:
            df_val = df[df['Velocidad'] == 50][val_params].dropna()
            
            if len(df_val) > 5:
                col4A_st, col4B_st = st.columns(2)
                with col4A_st:
                    st.markdown("**Matriz de Correlación Expandida**")
                    corr_matrix = df_val.corr()
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu', center=0, vmin=-1, vmax=1, fmt=".2f", linewidths=0.5, ax=ax_corr)
                    st.pyplot(fig_corr)
                    
                with col4B_st:
                    st.markdown("**Análisis de Componentes Principales (PCA)**")
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(df_val)
                    pca = PCA(n_components=2)
                    pca.fit(X_scaled)
                    loadings = pca.components_.T
                    
                    fig_pca, ax_pca = plt.subplots(figsize=(10, 8))
                    ax_pca.set_xlim(-1, 1)
                    ax_pca.set_ylim(-1, 1)
                    ax_pca.axhline(0, color='grey', linestyle='--', alpha=0.5)
                    ax_pca.axvline(0, color='grey', linestyle='--', alpha=0.5)
                    ax_pca.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Varianza)")
                    ax_pca.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Varianza)")
                    
                    for i, feature in enumerate(val_params):
                        ax_pca.arrow(0, 0, loadings[i, 0], loadings[i, 1], head_width=0.04, head_length=0.04, fc='red', ec='red', alpha=0.8)
                        ax_pca.text(loadings[i, 0]*1.12, loadings[i, 1]*1.12, feature, color='black', ha='center', va='center', fontsize=9)
                        
                    ax_pca.grid(True, linestyle=':', alpha=0.6)
                    st.pyplot(fig_pca)
            else:
                st.warning("No hay suficientes datos válidos a 50 mm/min para ejecutar la validación estadística.")
        else:
            st.error("Faltan columnas para este análisis.")
    else:
        st.error(f"Faltan columnas en el archivo Excel necesarias para generar este gráfico.")


# --- PESTAÑA 5: ÍNDICE DE FISURACIÓN ---
with tab_fisuracion:
    st.header("5. Índice de Fisuración (Experimental)")
    st.markdown("Espacio de pruebas paramétrico. Evalúa la relación directa entre la energía inicial, la capacidad de carga máxima y la penalización por la fragilidad post-pico.")
    st.markdown("⚠️ *Análisis filtrado únicamente para ensayos a **50 mm/min**.*")
    
    if 'Indice_Prepico' in df.columns:
        df_tab_lab = df[df['Velocidad'] == 50].copy()
        
        st.markdown("### Selecciona el modelo a evaluar:")
        tipo_indice = st.radio("Fórmula Base: (Selección × Tensión de Rotura) / |m75|", 
                               ["Energía Pre-pico", "Energía Post-pico"],
                               horizontal=True, label_visibility="collapsed")
        
        if "Pre-pico" in tipo_indice:
            col_idx = 'Indice_Prepico'
            col_num = 'Num_Prepico'
            col_den = 'abs_m75'
            col_energia = 'Gf_prepico'
            lbl_energia = 'Energía Pre-pico (J/m²)'
            lbl_num = 'Tensión (Pa) × Gf_prepico (J/m²)'
            lbl_idx = 'Índice de Fisuración (Pa)'
            lbl_den = '|m75| (N/m)'
            fmt_texto = '.0f'
        else: 
            col_idx = 'Indice_Postpico'
            col_num = 'Num_Postpico'
            col_den = 'abs_m75'
            col_energia = 'Gf_postpico'
            lbl_energia = 'Energía Post-pico (J/m²)'
            lbl_num = 'Tensión (Pa) × Gf_postpico (J/m²)'
            lbl_idx = 'Índice de Fisuración (Pa)'
            lbl_den = '|m75| (N/m)'
            fmt_texto = '.0f'
            
        st.markdown("---")
        
        st.subheader("5.1. Comparativa del Índice seleccionado por Mezcla")
        
        col_labA, col_labB = st.columns(2)
        with col_labA:
            agrup_x_lab = st.selectbox("Eje X:", ['Todas', 'Mezcla', 'Envejecimiento', 'RAP'], key="lab_x", index=0)
        with col_labB:
            color_lab = st.selectbox("Color:", ['Todas', 'Envejecimiento', 'RAP', 'Mezcla'], key="lab_color", index=0)
            
        agrupacion_lab = [agrup_x_lab] if agrup_x_lab == color_lab else [agrup_x_lab, color_lab]
        df_barras_lab = df_tab_lab.dropna(subset=[col_idx]).groupby(agrupacion_lab)[col_idx].mean().reset_index()
        df_barras_lab[color_lab] = df_barras_lab[color_lab].astype(str)
        
        modo_barra_lab = 'relative' if agrup_x_lab == color_lab else 'group'
        
        fig_bar_lab = px.bar(
            df_barras_lab, x=agrup_x_lab, y=col_idx, color=color_lab, 
            barmode=modo_barra_lab, text_auto=fmt_texto, color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_bar_lab.update_layout(height=450, xaxis_tickangle=-45, yaxis_title=lbl_idx)
        st.plotly_chart(fig_bar_lab, use_container_width=True)
        
        st.markdown("---")
        
        col_labC, col_labD = st.columns(2)
        
        with col_labC:
            st.subheader(f"5.2. Análisis 2D Específico")
            st.markdown(f"**Eje Y:** {lbl_num} | **Eje X:** {lbl_den}")
            
            df_valid_lab1 = df_tab_lab.dropna(subset=[col_num, col_den])
            
            tab_2d_1, tab_2d_2, tab_2d_3 = st.tabs(["📌 Solo Control", "♻️ Efecto RAP (AC22)", "⏳ Efecto Envejecimiento"])
            
            with tab_2d_1:
                df_ctrl = df_valid_lab1[df_valid_lab1['Envejecimiento'] == 'Control']
                fig_ctrl = px.scatter(
                    df_ctrl, x=col_den, y=col_num, color='Mezcla', 
                    hover_data=['Probeta', 'RAP'],
                    labels={col_den: lbl_den, col_num: ""},
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                fig_ctrl.update_layout(height=450, margin=dict(t=20))
                st.plotly_chart(fig_ctrl, use_container_width=True)
                
            with tab_2d_2:
                df_ac22 = df_valid_lab1[df_valid_lab1['Mezcla'] == 'AC22']
                fig_rap = px.scatter(
                    df_ac22, x=col_den, y=col_num, color='RAP', symbol='Envejecimiento',
                    hover_data=['Probeta'],
                    labels={col_den: lbl_den, col_num: ""},
                    color_discrete_sequence=px.colors.qualitative.Dark2
                )
                fig_rap.update_layout(height=450, margin=dict(t=20))
                st.plotly_chart(fig_rap, use_container_width=True)
                
            with tab_2d_3:
                df_env = df_valid_lab1[df_valid_lab1['RAP'] == 'Sin_RAP']
                fig_env = px.scatter(
                    df_env, x=col_den, y=col_num, color='Mezcla', symbol='Envejecimiento',
                    hover_data=['Probeta'],
                    labels={col_den: lbl_den, col_num: ""},
                    category_orders={"Envejecimiento": ["Control", "E1", "E2"]},
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                fig_env.update_layout(height=450, margin=dict(t=20))
                st.plotly_chart(fig_env, use_container_width=True)
            
        with col_labD:
            st.subheader("5.3. Nuevo Índice vs CT-Index")
            st.markdown("Comprueba si esta variante muestra correlación o tendencias similares al índice normativo.")
            
            df_valid_lab2 = df_tab_lab.dropna(subset=[col_idx, 'CT_Index'])
            fig_scatter_lab2 = px.scatter(
                df_valid_lab2, 
                x='CT_Index', y=col_idx, 
                color='Mezcla', 
                hover_data=['Probeta', 'Envejecimiento', 'RAP'],
                trendline="ols",
                labels={col_idx: lbl_idx, 'CT_Index': 'CT-Index'},
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig_scatter_lab2.update_layout(height=520)
            st.plotly_chart(fig_scatter_lab2, use_container_width=True)

        st.markdown("---")
        
        col_labE, col_labF = st.columns(2)
        
        with col_labE:
            st.subheader(f"5.4. Tensión / Fragilidad vs {col_energia}")
            st.markdown("Análisis de la tensión penalizada por la fragilidad frente a la energía analizada.")
            
            df_valid_lab3 = df_tab_lab.dropna(subset=['TR_sobre_m75', col_energia])
            fig_scatter_lab3 = px.scatter(
                df_valid_lab3, 
                x=col_energia, y='TR_sobre_m75', 
                color='Mezcla', 
                symbol='Envejecimiento',
                hover_data=['Probeta', 'RAP'],
                labels={col_energia: lbl_energia, 'TR_sobre_m75': 'Tensión Rotura / |m75|'},
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            
            min_x3, max_x3 = df_valid_lab3[col_energia].min(), df_valid_lab3[col_energia].max()
            min_y3, max_y3 = df_valid_lab3['TR_sobre_m75'].min(), df_valid_lab3['TR_sobre_m75'].max()
            fig_scatter_lab3.add_shape(type="line", x0=min_x3, y0=min_y3, x1=max_x3, y1=max_y3, line=dict(color="rgba(150,150,150,0.5)", dash="dash", width=2), layer="below")
            
            fig_scatter_lab3.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
            fig_scatter_lab3.update_layout(height=500)
            st.plotly_chart(fig_scatter_lab3, use_container_width=True)

        with col_labF:
            st.subheader(f"5.5. Tensión vs Fragilidad (Burbuja = {col_energia})")
            st.markdown("Relación 2D donde el tamaño del punto representa la energía analizada.")
            
            df_valid_lab4 = df_tab_lab.dropna(subset=['Tension_Rotura_SI', 'abs_m75', col_energia]).copy()
            df_valid_lab4['Tamano_Visual'] = df_valid_lab4[col_energia] ** 3
            
            fig_scatter_lab4 = px.scatter(
                df_valid_lab4, 
                x='abs_m75', y='Tension_Rotura_SI', 
                size='Tamano_Visual',
                color='Mezcla', 
                symbol='Envejecimiento',
                hover_data={'Tamano_Visual': False, col_energia: True, 'Probeta': True, 'RAP': True},
                labels={'abs_m75': '|m75| (N/m)', 'Tension_Rotura_SI': 'Tensión de Rotura (Pa)'},
                color_discrete_sequence=px.colors.qualitative.Set1,
                size_max=14 
            )
            
            fig_scatter_lab4.update_traces(marker=dict(sizemin=2, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
            min_x4, max_x4 = df_valid_lab4['abs_m75'].min(), df_valid_lab4['abs_m75'].max()
            min_y4, max_y4 = df_valid_lab4['Tension_Rotura_SI'].min(), df_valid_lab4['Tension_Rotura_SI'].max()
            fig_scatter_lab4.add_shape(type="line", x0=min_x4, y0=min_y4, x1=max_x4, y1=max_y4, line=dict(color="rgba(150,150,150,0.5)", dash="dash", width=2), layer="below")
            
            fig_scatter_lab4.update_layout(height=500)
            st.plotly_chart(fig_scatter_lab4, use_container_width=True)

        st.markdown("---")
        
        st.header("5.6. Análisis de Durabilidad: Retención de Fisuración")
        st.markdown("""
        Plantear un diagrama de espacio de diseño entre el estado inicial y la tasa de retención al envejecer es una herramienta fundamental en metodologías como el Diseño Equilibrado (BMD).

        * **Eje X:** Índice de Fisuración Inicial ($IF_{Control}$). Representa la calidad base tras la compactación.
        * **Eje Y:** Tasa de Retención ($IF_{E2} / IF_{Control} \cdot 100$). Representa qué porcentaje de sus propiedades sobrevive a la vida útil.

        **Zonas de Diagnóstico (Línea umbral en el 70%):**
        * **Zona Óptima (Arriba a la derecha):** El objetivo del diseño. Alto índice de fisuración inicial y retención superior al 70% frente al envejecimiento.
        * **Zona de Colapso (Abajo a la derecha):** Falsos positivos. Mezclas con un altísimo IF inicial, pero que se degradan brutalmente con la oxidación (retención pobre).
        * **Zona de Estabilidad Rígida (Arriba a la izquierda):** Retienen bien sus propiedades (estables en el tiempo), pero parten de una base muy frágil. Típico de mezclas duras muy oxidadas en su concepción.
        * **Zona de Riesgo (Abajo a la izquierda):** Nacen frágiles y además el envejecimiento destruye la escasa matriz que les queda.
        """)
        
        df_dur = df_tab_lab.groupby(['Mezcla', 'RAP', 'Envejecimiento'])[col_idx].mean().reset_index()
        try:
            df_pivot = df_dur.pivot(index=['Mezcla', 'RAP'], columns='Envejecimiento', values=col_idx).reset_index()
            
            if 'Control' in df_pivot.columns and 'E2' in df_pivot.columns:
                df_pivot['Tasa_Retencion_E2 (%)'] = (df_pivot['E2'] / df_pivot['Control']) * 100
                
                col_durA, col_durB = st.columns([1.5, 1])
                
                with col_durA:
                    fig_dur = px.scatter(
                        df_pivot, x='Control', y='Tasa_Retencion_E2 (%)', 
                        color='Mezcla', symbol='RAP',
                        hover_data=['Mezcla', 'RAP', 'E2'],
                        labels={'Control': f'{lbl_idx} (Control)', 'Tasa_Retencion_E2 (%)': 'Tasa de Retención E2 (%)'},
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )
                    fig_dur.update_traces(marker=dict(size=14, line=dict(width=1, color='DarkSlateGrey')))
                    
                    # Línea de umbral en el 70% de retención
                    min_x_dur = df_pivot['Control'].min() * 0.8
                    max_x_dur = df_pivot['Control'].max() * 1.2
                    fig_dur.add_shape(type="line", x0=min_x_dur, y0=70, x1=max_x_dur, y1=70, 
                                      line=dict(color="red", dash="dash", width=2), layer="below")
                    
                    # Forzar a que el eje Y empiece en 0 explícitamente y suba hasta al menos el 100%
                    max_y_dur = df_pivot['Tasa_Retencion_E2 (%)'].max()
                    max_y_dur = max(100, max_y_dur * 1.1) if pd.notna(max_y_dur) else 100
                    
                    fig_dur.update_layout(
                        height=500, 
                        title="Espacio de Diseño de Durabilidad (Umbral: 70%)",
                        yaxis=dict(range=[0, max_y_dur])
                    )
                    st.plotly_chart(fig_dur, use_container_width=True)
                    
                with col_durB:
                
                    st.markdown("**Tabla de Diagnóstico**")
                    df_show = df_pivot.copy()
                    
                    cols_show = ['Mezcla', 'RAP', 'Control']
                    if 'E1' in df_show.columns: cols_show.append('E1')
                    cols_show.append('E2')
                    cols_show.append('Tasa_Retencion_E2 (%)')
                    
                    df_show = df_show[cols_show]
                    
                    format_dict = {'Control': '{:.0f}', 'E2': '{:.0f}', 'Tasa_Retencion_E2 (%)': '{:.1f}%'}
                    if 'E1' in df_show.columns:
                        format_dict['E1'] = '{:.0f}'
                    
                    st.dataframe(df_show.style.format(format_dict, na_rep="-"))
                    
            else:
                st.warning("Faltan datos en estado 'Control' o 'E2' en las series para poder trazar el ciclo completo de durabilidad.")
        except Exception as e:
            st.warning(f"No se ha podido computar el análisis de durabilidad debido a la estructura actual de los datos: {e}")
            
    else:
        st.error("No se puede calcular el nuevo índice. Faltan datos necesarios en el archivo.")

# --- PESTAÑA 6: PODER DISCRIMINATORIO ---
with tab_disc:
    st.header("6. Análisis de Poder Discriminatorio")
    st.markdown("""
    Para identificar estadísticamente qué parámetros tienen el mayor poder discriminatorio, este módulo emplea dos metodologías de Machine Learning y estadística avanzada aplicadas a los ensayos de **50 mm/min**:
    1. **ANOVA de una vía**: Filtro estadístico que calcula el Valor F (varianza entre grupos vs dentro de los grupos). A mayor F, mayor poder separador aislado.
    2. **Random Forest (Feature Importance)**: Algoritmo de árboles de decisión que cuantifica qué peso porcentual tiene cada variable en la clasificación multivariante.
    """)
    st.markdown("---")
    
    col_tgt1, col_tgt2 = st.columns(2)
    with col_tgt1:
        target_col = st.radio("Selecciona la variable objetivo a clasificar:", ["Mezcla", "Envejecimiento"], horizontal=True)
    
    df_ml = df[df['Velocidad'] == 50].copy()
    
    with col_tgt2:
        if target_col == "Envejecimiento":
            st.info("💡 Sugerencia: Al evaluar el envejecimiento, es mejor filtrar por una mezcla concreta para evitar que las diferencias estructurales enmascaren el efecto térmico.")
            mezclas_ml = df_ml['Mezcla'].dropna().unique().tolist()
            filtro_mezcla_ml = st.selectbox("Filtrar datos por Mezcla:", ["Todas juntas"] + mezclas_ml)
            if filtro_mezcla_ml != "Todas juntas":
                df_ml = df_ml[df_ml['Mezcla'] == filtro_mezcla_ml]
        else:
            st.info("💡 Para clasificar mezclas, el algoritmo necesita verlas todas simultáneamente. Si lo deseas, puedes aislar un estado de envejecimiento concreto.")
            env_ml = df_ml['Envejecimiento'].dropna().unique().tolist()
            filtro_env_ml = st.selectbox("Filtrar datos por Envejecimiento:", ["Todos juntos"] + env_ml)
            if filtro_env_ml != "Todos juntos":
                df_ml = df_ml[df_ml['Envejecimiento'] == filtro_env_ml]
    
    features_obj = ['Carga_pico', 'Gf_prepico', 'Gf_postpico', 'abs_m75', 'l75', 'lpeak', 'Tension_Rotura']
    valid_features = [f for f in features_obj if f in df_ml.columns]
    
    df_ml = df_ml.dropna(subset=valid_features + [target_col])
    
    if len(df_ml[target_col].unique()) > 1 and len(df_ml) >= 5:
        X = df_ml[valid_features]
        y = df_ml[target_col]
        
        col_discA, col_discB = st.columns(2)
        
        with col_discA:
            st.subheader("6.1. Ranking Univariante (ANOVA F-Value)")
            f_dict = {}
            for col in valid_features:
                groups = [group[col].values for name, group in df_ml.groupby(target_col) if len(group) > 0]
                if len(groups) > 1:
                    f_stat, p_val = f_oneway(*groups)
                    f_dict[col] = f_stat
            
            if f_dict:
                df_f = pd.DataFrame(list(f_dict.items()), columns=['Parámetro', 'F-Value']).sort_values('F-Value', ascending=False)
                fig_anova = px.bar(df_f, x='F-Value', y='Parámetro', orientation='h', text_auto='.1f', color='F-Value', color_continuous_scale='Blues')
                fig_anova.update_layout(yaxis={'categoryorder':'total ascending'}, height=450, coloraxis_showscale=False)
                st.plotly_chart(fig_anova, use_container_width=True)
            else:
                st.warning("No se pudo calcular el ANOVA para estos grupos.")
        
        with col_discB:
            st.subheader("6.2. Importancia Relativa (Random Forest)")
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            df_rf = pd.DataFrame({'Parámetro': valid_features, 'Importancia (%)': rf.feature_importances_ * 100}).sort_values('Importancia (%)', ascending=False)
            fig_rf = px.bar(df_rf, x='Importancia (%)', y='Parámetro', orientation='h', text_auto='.1f', color='Importancia (%)', color_continuous_scale='Oranges')
            fig_rf.update_layout(yaxis={'categoryorder':'total ascending'}, height=450, coloraxis_showscale=False)
            st.plotly_chart(fig_rf, use_container_width=True)
            
    else:
        st.warning("No hay suficientes datos válidos (sin celdas vacías) o no hay suficientes categorías distintas para entrenar los modelos matemáticos en esta selección.")