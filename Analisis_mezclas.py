import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# ---------------------------------------------------------
# 1. CONFIGURACIÓN E INTERFAZ
# ---------------------------------------------------------
st.set_page_config(page_title="DURAPAV - Análisis Factorial", layout="wide")

st.sidebar.markdown("### Laboratorio de Caminos - UPV")
st.sidebar.markdown("**Proyecto DURAPAV**")
st.sidebar.markdown("---")

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

df_original = cargar_datos()

# --- FILTRO GLOBAL (SIDEBAR) ---
st.sidebar.markdown("### 🎛️ Filtro Principal")
st.sidebar.markdown("Selecciona la mezcla para aislar su análisis en todo el dashboard.")

mezclas_disponibles = df_original['Mezcla'].dropna().unique().tolist()
mezcla_global = st.sidebar.selectbox(
    "Filtro de Mezcla:", 
    ["Todas"] + mezclas_disponibles
)

# Aplicamos el filtro al dataframe que usará el resto de la aplicación
if mezcla_global != "Todas":
    df = df_original[df_original['Mezcla'] == mezcla_global].copy()
else:
    df = df_original.copy()

st.sidebar.markdown("---")

st.title("Análisis Dinámico de Parámetros: Ensayo Ideal-CT")
st.markdown(f"**Modo de visualización actual:** Explorando datos de ➡️ **{mezcla_global}**")

# --- DESCRIPCIÓN INICIAL ---
with st.expander("Ver descripción de variables y parámetros del ensayo"):
    col_desc1, col_desc2 = st.columns(2)
    with col_desc1:
        st.info("""
        **Variables empleadas como filtros**
        * **Envejecimiento:** Sin_Envejecer (E0); Nivel_1 (ej. 2 días a 85 ºC); Nivel_2 (ej. 5 días a 85 ºC).
        * **RAP:** Sin_RAP = mezcla sin asfalto recuperado; 30_RAP = mezcla con 30% de asfalto recuperado.
        * **Velocidad:** velocidad de desplazamiento del ensayo, con valores de 1, 2 y 50 mm/min.
        """)
    with col_desc2:
        st.info("""
        **Parámetros del ensayo Ideal-CT**
        * **Gf:** tenacidad o energía de fractura (J/m²).
        * **Gf prepico y postpico:** contribución energética antes y después de la carga máxima.
        * **l75:** ductilidad, asociada a la capacidad de deformación.
        * **m75 / m_xx:** fragilidad, asociada a la pendiente postpico y propagación de fisura.
        * **Carga Pico:** resistencia asociada al inicio de la formación de la fisura.
        * **CT_Index:** índice global de tolerancia a la fisuración.
        """)

st.markdown("---")

# ---------------------------------------------------------
# 2. PESTAÑAS DE NAVEGACIÓN
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Gráfico de Barras", 
    "Radar", 
    "Gráfico de Cajas",
    "Regresión de Rigidez",
    "Reparto Energético"
])

opciones_var = ['Carga_pico', 'CT_Index', 'Gf', 'm75', 'l75']
if 'Tension_Rotura' in df.columns: opciones_var.append('Tension_Rotura')

# --- PESTAÑA 1: GRÁFICO DE BARRAS ---
with tab1:
    st.header("1. Comparativa Directa por Parámetro (Medias)")
    
    col1A, col1B, col1C = st.columns(3)
    with col1A:
        param_bar = st.selectbox("Parámetro (Eje Y):", opciones_var, key="bar_y")
    with col1B:
        eje_x_bar = st.selectbox("Agrupar por (Eje X):", ['Todas', 'Envejecimiento', 'RAP', 'Velocidad', 'Mezcla'], key="bar_x")
    with col1C:
        color_bar = st.selectbox("Separar colores por:", ['Velocidad', 'Envejecimiento', 'RAP', 'Mezcla'], key="bar_color")

    df_barras = df.groupby([eje_x_bar, color_bar])[param_bar].mean().reset_index()
    
    fig_bar = px.bar(
        df_barras, x=eje_x_bar, y=param_bar, color=color_bar, 
        barmode='group', text_auto='.2f', color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_bar.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)

# --- PESTAÑA 2: RADAR DE COMPORTAMIENTO (FILTRADO) ---
with tab2:
    st.header("2. Huella Mecánica (Normalizada)")
    st.markdown("⚠️ *Este radar muestra **únicamente** los ensayos realizados a una velocidad de **50 mm/min**.*")
    
    df_50 = df[df['Velocidad'] == 50].copy()
    
    variables_radar = st.multiselect(
        "Parámetros del radar:",
        ['Carga_pico', 'Gf', 'CT_Index', 'Rigidez_20', 'm75', 'l75'],
        default=['Carga_pico', 'Gf', 'CT_Index', 'm75', 'l75']
    )
    
    if len(variables_radar) > 2 and not df_50.empty:
        df_agrupado = df_50.groupby('Todas')[variables_radar].mean().reset_index()
        
        df_radar = df_agrupado.copy()
        for col in variables_radar:
            val_min = df_radar[col].min()
            val_max = df_radar[col].max()
            if val_max != val_min:
                df_radar[col] = (df_radar[col] - val_min) / (val_max - val_min)
            else:
                df_radar[col] = 1.0

        fig_radar = go.Figure()
        for i, row in df_radar.iterrows():
            valores = row[variables_radar].tolist()
            valores += [valores[0]] 
            ejes = variables_radar + [variables_radar[0]]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=valores, theta=ejes, fill='toself', name=row['Todas']
            ))
            
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, height=600)
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.warning("Selecciona al menos 3 parámetros o comprueba que hay datos a 50 mm/min para la mezcla seleccionada.")

# --- PESTAÑA 3: GRÁFICO DE CAJAS ---
with tab3:
    st.header("3. Análisis de Dispersión y Variabilidad")
    st.markdown("Este gráfico muestra todos los resultados individuales. Las cajas representan dónde se agrupa el 50% de las probetas, la línea central es la mediana, y los puntos sueltos son valores atípicos (*outliers*).")
    
    col3A, col3B, col3C = st.columns(3)
    with col3A:
        param_box = st.selectbox("Parámetro (Eje Y):", opciones_var, key="box_y")
    with col3B:
        eje_x_box = st.selectbox("Agrupar por (Eje X):", ['Todas', 'Envejecimiento', 'RAP', 'Velocidad', 'Mezcla'], key="box_x")
    with col3C:
        color_box = st.selectbox("Separar colores por:", ['Velocidad', 'RAP', 'Envejecimiento', 'Mezcla'], key="box_color")

    fig_box = px.box(
        df, x=eje_x_box, y=param_box, color=color_box, 
        points="all", 
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_box.update_layout(height=600, xaxis_tickangle=-45)
    st.plotly_chart(fig_box, use_container_width=True)

# --- PESTAÑA 4: ANÁLISIS DE RIGIDEZ Y CORRELACIONES ---
with tab4:
    st.header("4. Análisis de Regresión: Modelización de la Rigidez")
    st.markdown("⚠️ *Este análisis excluye los ensayos a bajas velocidades, mostrando **solo 50 mm/min** para el modelo.*")
    
    df_est = df[df['Velocidad'] == 50].copy()
        
    st.markdown("---")
    
    # RIGIDEZ VS PENDIENTE
    st.subheader("4.1. Influencia de la Pendiente Pre-Pico")
    
    lista_pendientes = [
        'm30_10', 'm35_10', 'm40_10', 'm45_10', 'm50_10', 'm55_10', 'm60_10', 'm65_10', 'm70_10', 
        'm30_20', 'm35_20', 'm40_20', 'm45_20', 'm50_20', 'm55_20', 'm60_20', 'm65_20', 'm70_20'
    ]
    pendientes_validas = [p for p in lista_pendientes if p in df_est.columns]
    
    col_4A, col_4B = st.columns([1, 2])
    with col_4A:
        if pendientes_validas:
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
        else:
            st.warning("No se encontraron columnas de pendientes.")
            
    with col_4B:
        if pendientes_validas and len(df_clean) > 2:
            fig_p, ax_p = plt.subplots(figsize=(8, 4))
            sns.scatterplot(data=df_clean, x=pend_selec, y='Rigidez_20', color='#3498db', s=80, alpha=0.7, ax=ax_p)
            sns.regplot(data=df_clean, x=pend_selec, y='Rigidez_20', scatter=False, color='#e74c3c', line_kws={'linestyle': '--'}, ax=ax_p)
            ax_p.set_ylabel('Rigidez Real (MPa)')
            ax_p.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig_p)

    st.markdown("---")
    
    # RIGIDEZ VS CARGA Y TENSIÓN
    st.subheader("4.2. Parámetros de Rotura vs Rigidez")
    col_4C, col_4D = st.columns(2)
    
    with col_4C:
        st.markdown("**Carga Pico**")
        df_carga = df_est.dropna(subset=['Carga_pico', 'Rigidez_20'])
        if len(df_carga) > 2:
            X_carga = sm.add_constant(df_carga['Carga_pico'])
            mod_c = sm.OLS(df_carga['Rigidez_20'], X_carga).fit()
            
            fig_c, ax_c = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=df_carga, x='Carga_pico', y='Rigidez_20', color='#2ecc71', s=70, alpha=0.7, ax=ax_c)
            sns.regplot(data=df_carga, x='Carga_pico', y='Rigidez_20', scatter=False, color='#e74c3c', ax=ax_c)
            ax_c.set_xlabel('Carga Pico (N/kN)')
            ax_c.set_ylabel('Rigidez (MPa)')
            ax_c.grid(True, alpha=0.3)
            st.pyplot(fig_c)
            st.caption(f"R²: {mod_c.rsquared:.4f} | p-value: {mod_c.pvalues['Carga_pico']:.4f}")
            
    with col_4D:
        st.markdown("**Tensión de Rotura**")
        if 'Tension_Rotura' in df_est.columns:
            df_tens = df_est.dropna(subset=['Tension_Rotura', 'Rigidez_20'])
            if len(df_tens) > 2:
                X_tens = sm.add_constant(df_tens['Tension_Rotura'])
                mod_t = sm.OLS(df_tens['Rigidez_20'], X_tens).fit()
                
                fig_t, ax_t = plt.subplots(figsize=(6, 4))
                sns.scatterplot(data=df_tens, x='Tension_Rotura', y='Rigidez_20', color='#9b59b6', s=70, alpha=0.7, ax=ax_t)
                sns.regplot(data=df_tens, x='Tension_Rotura', y='Rigidez_20', scatter=False, color='#e74c3c', ax=ax_t)
                ax_t.set_xlabel('Tensión Rotura (MPa)')
                ax_t.set_ylabel('Rigidez (MPa)')
                ax_t.grid(True, alpha=0.3)
                st.pyplot(fig_t)
                st.caption(f"R²: {mod_t.rsquared:.4f} | p-value: {mod_t.pvalues['Tension_Rotura']:.4f}")

    st.markdown("---")
    
    # MAPA DE CALOR
    st.subheader("4.3. Matriz de Correlación Global")
    
    cols_heatmap = ['Rigidez_20', 'Carga_pico']
    if 'Tension_Rotura' in df_est.columns:
        cols_heatmap.append('Tension_Rotura')
    cols_heatmap.extend(pendientes_validas)
    
    if len(df_est) > 2:
        matriz_corr = df_est[cols_heatmap].corr()
        fig_heat, ax_heat = plt.subplots(figsize=(18, 10))
        sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax_heat, annot_kws={"size": 8})
        st.pyplot(fig_heat)

# --- PESTAÑA 5: REPARTO ENERGÉTICO ---
with tab5:
    st.header("5. Balance Energético de Fractura")
    st.markdown("Análisis de la energía para iniciar la fisuración (**Gf prepico**) vs resistencia residual (**Gf postpico**).")
    
    if 'Gf_prepico' in df.columns and 'Gf_postpico' in df.columns:
        col5A, col5B, col5C = st.columns(3)
        with col5A:
            eje_x_energia = st.selectbox("Agrupar por (Eje X):", ['Todas', 'Envejecimiento', 'RAP', 'Mezcla'], key="energia_x")
        with col5B:
            filtro_vel = st.multiselect(
                "Filtrar velocidades (mm/min):", 
                options=sorted(df['Velocidad'].dropna().unique().tolist()),
                default=sorted(df['Velocidad'].dropna().unique().tolist()),
                key="energia_vel"
            )
        with col5C:
            tipo_vista = st.radio("Modo de representación:", ["Valores Absolutos (J/m²)", "Porcentaje Relativo (100%)"], horizontal=True, key="energia_modo")
            
        df_energia = df[df['Velocidad'].isin(filtro_vel)].copy()
        
        if not df_energia.empty:
            df_g_mean = df_energia.groupby(eje_x_energia)[['Gf_prepico', 'Gf_postpico']].mean().reset_index()
            
            if tipo_vista == "Porcentaje Relativo (100%)":
                total_gf = df_g_mean['Gf_prepico'] + df_g_mean['Gf_postpico']
                df_g_mean['Gf_prepico'] = (df_g_mean['Gf_prepico'] / total_gf) * 100
                df_g_mean['Gf_postpico'] = (df_g_mean['Gf_postpico'] / total_gf) * 100
                formato_texto = '.1f'
                eje_y_titulo = "Porcentaje de Energía Total (%)"
            else:
                formato_texto = '.1f'
                eje_y_titulo = "Energía de Fractura (J/m²)"
                
            df_melted = pd.melt(df_g_mean, id_vars=[eje_x_energia], value_vars=['Gf_prepico', 'Gf_postpico'], var_name='Fase_Energia', value_name='Valor')
            
            fig_stack = px.bar(
                df_melted, x=eje_x_energia, y='Valor', color='Fase_Energia', barmode='stack', text_auto=formato_texto,
                color_discrete_map={'Gf_prepico': '#3498db', 'Gf_postpico': '#e67e22'}
            )
            fig_stack.update_layout(height=550, xaxis_tickangle=-45, yaxis_title=eje_y_titulo)
            st.plotly_chart(fig_stack, use_container_width=True)
        else:
            st.warning("Selecciona al menos una velocidad.")
    else:
        st.error("No se han encontrado las columnas 'Gf_prepico' y 'Gf_postpico'.")