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

# ---------------------------------------------------------
# 1. CONFIGURACIÓN E INTERFAZ
# ---------------------------------------------------------
st.set_page_config(page_title="DURAPAV - Análisis Factorial", layout="wide")

st.sidebar.markdown("### Laboratorio de Caminos - UPV")
st.sidebar.markdown("**Proyecto DURAPAV**")
st.sidebar.markdown("---")

st.title("Análisis Dinámico de Parámetros: Ensayo Ideal-CT")

# --- DESCRIPCIÓN INICIAL ---
st.markdown("El dashboard permite explorar gráficamente los resultados obtenidos mediante el ensayo Ideal-CT. Su objetivo es facilitar la comparación entre mezclas y condiciones de ensayo, evaluando cómo cambian la tenacidad, la ductilidad, la fragilidad y la resistencia en función del envejecimiento, la incorporación de Asfalto Recuperado (RA) y la velocidad de ensayo.")

col_desc1, col_desc2 = st.columns(2)
with col_desc1:
    st.info("""
    **Variables empleadas como filtros**
    * **Envejecimiento:** E0 (Sin_Envejecer); E1 (2 días a 85 °C); E2 (5 días a 85 °C).
    * **RAP:** Sin_RAP = mezcla sin asfalto recuperado; RAP = mezcla con asfalto recuperado.
    * **Velocidad:** velocidad de desplazamiento del ensayo, con valores de 2 y 50 mm/min.
    * **Mezcla:** AC16, AC22, BBTM11, SMA16.
    """)
with col_desc2:
    st.info("""
    **Parámetros del ensayo Ideal-CT**
    * **Gf:** tenacidad o energía de fractura (J/m²).
    * **Gf prepico y postpico:** contribución energética antes y después de la carga máxima.
    * **l75:** ductilidad, asociada a la capacidad de deformación.
    * **m75:** fragilidad, asociada a la pendiente postpico y propagación de fisura.
    * **mxx_10/20:** pendiente prepico a xx% de la carga pico calculada en un rango de 10 o 20.
    * **Carga Pico/Tensión de rotura:** fuerza máxima asociada al inicio de la formación de la macrofisura.
    * **CT_Index:** índice global de tolerancia a la fisuración.
    * **Ratio_Flexibilidad:** l75 / m75.
    """)

@st.cache_data
def cargar_datos():
    df = pd.read_excel('datos.xlsx')
    df.columns = df.columns.astype(str).str.strip()
    
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    if 'Espesor' in df.columns and 'Diametro' in df.columns:
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
    # Multiplicamos por un millón (1e6) para escalar el índice a valores legibles
    df['Ratio_Flexibilidad'] = (df['l75'] / df['m75'].abs().replace(0, np.nan)) * 1e6
    df['abs_m75'] = df['m75'].abs()

st.markdown("---")

# ---------------------------------------------------------
# 2. PESTAÑAS DE NAVEGACIÓN
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Gráfico de Barras", 
    "Gráfico de Cajas",
    "Regresión de Rigidez",
    "Reparto Energético",
    "Diagrama Stiff",
    "Validación Estadística"
])

# --- PESTAÑA 1: GRÁFICO DE BARRAS ---
with tab1:
    st.header("1. Comparativa Directa por Parámetro")
    
    col1A, col1B, col1C = st.columns(3)
    with col1A:
        opciones_var = ['Carga_pico', 'CT_Index', 'Gf', 'm75', 'l75', 'Ratio_Flexibilidad']
        if 'Tension_Rotura' in df.columns: opciones_var.append('Tension_Rotura')
        param_bar = st.selectbox("Parámetro (Eje Y):", opciones_var, key="bar_y")
    with col1B:
        eje_x_bar = st.selectbox("Agrupar por (Eje X):", ['Todas', 'Mezcla', 'Envejecimiento', 'RAP'], key="bar_x")
    with col1C:
        color_bar = st.selectbox("Separar colores por:", ['Velocidad', 'Envejecimiento', 'RAP', 'Mezcla'], key="bar_color")

    df_barras = df.groupby([eje_x_bar, color_bar])[param_bar].mean().reset_index()
    df_barras[color_bar] = df_barras[color_bar].astype(str)
    
    fig_bar = px.bar(
        df_barras, x=eje_x_bar, y=param_bar, color=color_bar, 
        barmode='group', text_auto='.2f', color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_bar.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)

# --- PESTAÑA 2: GRÁFICO DE CAJAS ---
with tab2:
    st.header("2. Análisis de Dispersión y Variabilidad")
    st.markdown("Este gráfico muestra todos los resultados individuales. Las cajas representan dónde se agrupa el 50% de las probetas, la línea central es la mediana, y los puntos sueltos son valores atípicos (*outliers*).")
    
    col3A, col3B, col3C, col3D = st.columns(4)
    with col3A:
        mezclas_disp_t3 = df['Mezcla'].dropna().unique().tolist()
        filtro_mezcla_t3 = st.selectbox("Filtrar Mezcla:", ["Todas"] + mezclas_disp_t3, key="box_mezcla")
    with col3B:
        param_box = st.selectbox("Parámetro (Eje Y):", opciones_var, key="box_y")
    with col3C:
        eje_x_box = st.selectbox("Agrupar por (Eje X):", ['Todas', 'Mezcla', 'Envejecimiento', 'RAP'], key="box_x")
    with col3D:
        color_box = st.selectbox("Separar colores por:", ['Velocidad', 'RAP', 'Envejecimiento', 'Mezcla'], key="box_color")

    if filtro_mezcla_t3 != "Todas":
        df_cajas = df[df['Mezcla'] == filtro_mezcla_t3].copy()
    else:
        df_cajas = df.copy()

    df_cajas[color_box] = df_cajas[color_box].astype(str)

    fig_box = px.box(
        df_cajas, x=eje_x_box, y=param_box, color=color_box, 
        points="all", 
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_box.update_layout(height=600, xaxis_tickangle=-45)
    st.plotly_chart(fig_box, use_container_width=True)

# --- PESTAÑA 3: ANÁLISIS DE RIGIDEZ Y CORRELACIONES ---
with tab3:
    st.header("3. Análisis de Regresión: Modelización de la Rigidez")
    st.markdown("*Este análisis de regresión relaciona los parámetros con la rigidez a **20 °C** y se excluyen los ensayos a velocidad de **2 mm/min**.*")
    st.markdown("*Los colores de los puntos corresponden al tipo de mezcla y los números al indicador de cada probeta.*")
    
    df_tab4 = df[df['Velocidad'] == 50].copy()
    
    mezclas_unicas = df_tab4['Mezcla'].dropna().unique().tolist()
    mezcla_elegida = st.selectbox(
        "Filtra por mezcla:", 
        ['Todas'] + mezclas_unicas,
        help="Si eliges 'Todas', el modelo de regresión evaluará el comportamiento global de las probetas a 50 mm/min juntas."
    )
    
    if mezcla_elegida == 'Todas':
        df_est = df_tab4.copy()
    else:
        df_est = df_tab4[df_tab4['Mezcla'] == mezcla_elegida].copy()
        
    st.markdown("---")
    
    st.subheader("3.1. Ajuste Pendiente Pre-Pico y Rigidez ")
    lista_pendientes = [
        'm30_10', 'm35_10', 'm40_10', 'm45_10', 'm50_10', 'm55_10', 'm60_10', 'm65_10', 'm70_10', 
        'm30_20', 'm35_20', 'm40_20', 'm45_20', 'm50_20', 'm55_20', 'm60_20', 'm65_20', 'm70_20'
    ]
    pendientes_validas = [p for p in lista_pendientes if p in df_est.columns]
    
    col_4A, col_4B = st.columns([1, 2])
    with col_4A:
        pend_selec = st.selectbox("Selecciona la pendiente a evaluar y el rango:", pendientes_validas)
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
            
    with col_4B:
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
    
    st.subheader("3.2. Ajuste Pendiente Pre-Pico y los Parámetros de Rotura")
    col_4C, col_4D = st.columns(2)
    with col_4C:
        st.markdown("**Carga Pico**")
        df_carga = df_est.dropna(subset=['Carga_pico', 'Rigidez_20'])
        if len(df_carga) > 2:
            X_carga = sm.add_constant(df_carga['Carga_pico'])
            mod_c = sm.OLS(df_carga['Rigidez_20'], X_carga).fit()
            fig_c, ax_c = plt.subplots(figsize=(7, 5))
            sns.scatterplot(data=df_carga, x='Carga_pico', y='Rigidez_20', hue='Todas', palette='tab10', s=80, alpha=0.8, ax=ax_c)
            sns.regplot(data=df_carga, x='Carga_pico', y='Rigidez_20', scatter=False, color='black', line_kws={'linestyle': '--', 'alpha':0.6}, ax=ax_c)
            if 'Probeta' in df_carga.columns:
                for idx, row in df_carga.iterrows():
                    if pd.notna(row['Probeta']):
                        ax_c.annotate(str(row['Probeta']), (row['Carga_pico'], row['Rigidez_20']), textcoords="offset points", xytext=(5,5), ha='left', fontsize=8, alpha=0.8)
            ax_c.set_xlabel('Carga Pico (N/kN)')
            ax_c.set_ylabel('Rigidez (MPa)')
            ax_c.grid(True, alpha=0.3)
            ax_c.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
            fig_c.tight_layout()
            st.pyplot(fig_c)
            st.caption(f"R²: {mod_c.rsquared:.4f} | p-value: {mod_c.pvalues['Carga_pico']:.4f}")
            
    with col_4D:
        st.markdown("**Tensión de Rotura**")
        if 'Tension_Rotura' in df_est.columns:
            df_tens = df_est.dropna(subset=['Tension_Rotura', 'Rigidez_20'])
            if len(df_tens) > 2:
                X_tens = sm.add_constant(df_tens['Tension_Rotura'])
                mod_t = sm.OLS(df_tens['Rigidez_20'], X_tens).fit()
                fig_t, ax_t = plt.subplots(figsize=(7, 5))
                sns.scatterplot(data=df_tens, x='Tension_Rotura', y='Rigidez_20', hue='Todas', palette='tab10', s=80, alpha=0.8, ax=ax_t)
                sns.regplot(data=df_tens, x='Tension_Rotura', y='Rigidez_20', scatter=False, color='black', line_kws={'linestyle': '--', 'alpha':0.6}, ax=ax_t)
                if 'Probeta' in df_tens.columns:
                    for idx, row in df_tens.iterrows():
                        if pd.notna(row['Probeta']):
                            ax_t.annotate(str(row['Probeta']), (row['Tension_Rotura'], row['Rigidez_20']), textcoords="offset points", xytext=(5,5), ha='left', fontsize=8, alpha=0.8)
                ax_t.set_xlabel('Tensión Rotura (MPa)')
                ax_t.set_ylabel('Rigidez (MPa)')
                ax_t.grid(True, alpha=0.3)
                ax_t.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
                fig_t.tight_layout()
                st.pyplot(fig_t)
                st.caption(f"R²: {mod_t.rsquared:.4f} | p-value: {mod_t.pvalues['Tension_Rotura']:.4f}")

    st.markdown("---")
    st.subheader("3.3. Matrices de Correlación")
    st.info("""
    **¿Qué significan estos números?**
    Los valores mostrados representan el **Coeficiente de Correlación de Pearson ($r$)**. 
    Este coeficiente evalúa estadísticamente si existe una relación lineal entre dos variables.
    """)
    macro_params = ['Rigidez_20', 'Carga_pico']
    if 'Tension_Rotura' in df_est.columns:
        macro_params.append('Tension_Rotura')
        
    if len(df_est) > 2 and pendientes_validas:
        st.markdown("**A. Correlación cruzada: Parámetros Macro (Rigidez y Rotura) vs Pendientes pre-pico (y entre sí)**")
        corr_full = df_est[macro_params + pendientes_validas].corr()
        corr_macro_pendientes = corr_full.loc[macro_params, macro_params + pendientes_validas]
        fig_heat1, ax_heat1 = plt.subplots(figsize=(20, len(macro_params) * 1.5))
        sns.heatmap(corr_macro_pendientes, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=0.5, ax=ax_heat1, annot_kws={"size": 9})
        ax_heat1.tick_params(axis='y', rotation=0)
        st.pyplot(fig_heat1)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**B. Correlación entre las propias pendientes entre sí**")
        corr_pendientes = df_est[pendientes_validas].corr()
        mask = np.triu(np.ones_like(corr_pendientes, dtype=bool))
        fig_heat2, ax_heat2 = plt.subplots(figsize=(16, 12))
        sns.heatmap(corr_pendientes, mask=mask, annot=True, cmap='YlGnBu', fmt=".4f", linewidths=0.5, ax=ax_heat2, annot_kws={"size": 8})
        st.pyplot(fig_heat2)

# --- PESTAÑA 4: REPARTO ENERGÉTICO ---
with tab4:
    st.header("4. Balance Energético de Fractura")
    st.markdown("Análisis de la energía para iniciar la fisuración (**Gf prepico**) vs resistencia residual (**Gf postpico**).")
    if 'Gf_prepico' in df.columns and 'Gf_postpico' in df.columns:
        col5A, col5B, col5C = st.columns(3)
        with col5A:
            mezclas_disp = df['Mezcla'].dropna().unique().tolist()
            filtro_mezcla_t5 = st.selectbox("Elegir Mezcla:", ["Todas"] + mezclas_disp, key="t5_mezcla")
        with col5B:
            eje_x_energia = st.selectbox("Agrupar gráficas por (Eje X):", ["Todas", "Envejecimiento", "RAP", "Velocidad"], key="t5_var")
        with col5C:
            tipo_vista = st.radio("Modo de representación:", ["Valores Absolutos (J/m²)", "Porcentaje Relativo (100%)"], horizontal=False, key="energia_modo")
            
        st.markdown("---")
        if filtro_mezcla_t5 != "Todas":
            df_energia = df[df['Mezcla'] == filtro_mezcla_t5].copy()
        else:
            df_energia = df.copy()
            
        if not df_energia.empty:
            df_g_mean = df_energia.groupby(eje_x_energia)[['Gf_prepico', 'Gf_postpico']].mean().reset_index()
            if tipo_vista == "Porcentaje Relativo (100%)":
                total_gf = df_g_mean['Gf_prepico'] + df_g_mean['Gf_postpico']
                df_g_mean['Gf_prepico'] = (df_g_mean['Gf_prepico'] / total_gf) * 100
                df_g_mean['Gf_postpico'] = (df_g_mean['Gf_postpico'] / total_gf) * 100
                eje_y_titulo = "Porcentaje de Energía (%)"
            else:
                eje_y_titulo = "Energía (J/m²)"
                
            df_melted = pd.melt(df_g_mean, id_vars=[eje_x_energia], value_vars=['Gf_prepico', 'Gf_postpico'], var_name='Fase_Energia', value_name='Valor')
            fig_stack = px.bar(df_melted, x=eje_x_energia, y='Valor', color='Fase_Energia', barmode='stack', text_auto='.1f', color_discrete_map={'Gf_prepico': '#3498db', 'Gf_postpico': '#e67e22'})
            fig_stack.update_layout(height=550, xaxis_tickangle=-45, yaxis_title=eje_y_titulo)
            st.plotly_chart(fig_stack, use_container_width=True)

# --- PESTAÑA 5: DIAGRAMA STIFF (DIFERENCIAL) ---
with tab5:
    st.header("5. Firma Mecánica: Diagrama de Stiff Diferencial")
    st.markdown("Evalúa la **migración del centroide mecánico** al superponer dos estados de mezcla (ensayos a 50 mm/min). Desplazamientos hacia la izquierda indican rigidización/fragilidad, hacia la derecha indican ductilidad.")
    
    st.info("""
    **¿Cómo se construye la escala del diagrama?**
    Cada parámetro se normaliza en una escala de 0 a 1 utilizando sus valores mínimos y máximos globales de la serie de ensayos. 
    *   Los parámetros antagónicos asociados a **Rigidez/Fragilidad** se proyectan hacia el lado izquierdo (dibujados como valores de 0 a -1).
    *   Los parámetros asociados a **Ductilidad/Tenacidad** se proyectan hacia el lado derecho (dibujados como valores de 0 a +1).
    """)
    
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
                
                # Eje 1 (Y=3): Carga_pico (Izq) vs l75 (Der)
                # Eje 2 (Y=2): abs_m75 (Izq) vs Gf_postpico (Der)
                # Eje 3 (Y=1): Rigidez_20 (Izq) vs Ratio_Flexibilidad (Der)
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
                    zeroline=False, showgrid=True, gridcolor='rgba(200,200,200,0.2)',
                    title="Escala Normalizada (0 a 1)"
                ),
                yaxis=dict(range=[0.5, 3.5], showticklabels=False, zeroline=False, showgrid=False),
                annotations=list(fig_stiff.layout.annotations) + anotaciones_eje,
                height=650, margin=dict(l=20, r=20, t=40, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(fig_stiff, use_container_width=True)
        else:
            st.warning("No hay datos para alguna de las combinaciones seleccionadas. Por favor, cambia los filtros de RAP o Envejecimiento.")
    else:
        st.error(f"Faltan columnas en el archivo Excel necesarias para generar este gráfico: {', '.join(missing_cols)}")

# --- PESTAÑA 6: VALIDACIÓN ESTADÍSTICA (ACTUALIZADA) ---
with tab6:
    st.header("6. Justificación Estadística del Diagrama Stiff")
    st.markdown("Este apartado demuestra matemáticamente la correcta distribución de los ejes (Izquierda = Rigidez/Fragilidad vs Derecha = Ductilidad/Flexibilidad) utilizando los datos a **50 mm/min**.")
    
    val_params = ['Carga_pico', 'l75', 'Gf_prepico', 'Gf_postpico', 'Rigidez_20', 'm75', 'Ratio_Flexibilidad']
    missing_val_cols = [c for c in val_params if c not in df.columns]
    
    if not missing_val_cols:
        df_val = df[df['Velocidad'] == 50][val_params].dropna()
        
        if len(df_val) > 5:
            col6A, col6B = st.columns(2)
            
            with col6A:
                st.subheader("1. Matriz de Correlación Expandida")
                st.markdown("Verifica cómo las variables se correlacionan positivamente entre sí y negativamente con las variables opuestas del diagrama.")
                
                # 1. Calculamos la matriz de correlación
                corr_matrix = df_val.corr()
                
                # 2. Creamos una máscara para ocultar la diagonal superior
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                
                # 3. Generamos el gráfico aplicando la máscara
                fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu', center=0, vmin=-1, vmax=1, fmt=".2f", linewidths=0.5, ax=ax_corr)
                st.pyplot(fig_corr)
                
            with col6B:
                st.subheader("2. Análisis de Componentes Principales (PCA)")
                st.markdown("El PCA es un algoritmo estadístico que reduce la complejidad de los datos a un plano 2D, revelando visualmente qué variables se comportan de manera similar y cuáles son matemáticamente opuestas.")
                
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
        st.error(f"Faltan columnas para este análisis: {', '.join(missing_val_cols)}")