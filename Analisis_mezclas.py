import streamlit as st
import pandas as pd
import numpy as np
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
    * **Mezcla:** AC16, AC22, BBTM11.
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

# --- PESTAÑA 1: GRÁFICO DE BARRAS ---
with tab1:
    st.header("1. Comparativa Directa por Parámetro (Medias)")
    
    col1A, col1B, col1C = st.columns(3)
    with col1A:
        opciones_var = ['Carga_pico', 'CT_Index', 'Gf', 'm75', 'l75']
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
    
    if len(variables_radar) > 2:
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
        st.warning("Selecciona al menos 3 parámetros.")

# --- PESTAÑA 3: GRÁFICO DE CAJAS ---
with tab3:
    st.header("3. Análisis de Dispersión y Variabilidad")
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

    # Filtrar por la mezcla seleccionada
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

# --- PESTAÑA 4: ANÁLISIS DE RIGIDEZ Y CORRELACIONES ---
with tab4:
    st.header("4. Análisis de Regresión: Modelización de la Rigidez")
    st.markdown("⚠️ *Este análisis de regresión excluye los ensayos a bajas velocidades, mostrando únicamente los resultados a **50 mm/min**.*")
    st.markdown("*Los numeros de los gráficos de regresión son los indicadores de las probetas a las que corresponde cada punto.*")
    
    df_tab4 = df[df['Velocidad'] == 50].copy()
    
    mezclas_unicas = df_tab4['Mezcla'].dropna().unique().tolist()
    mezcla_elegida = st.selectbox(
        "Filtra los datos del modelo estadístico:", 
        ['Todas'] + mezclas_unicas,
        help="Si eliges 'Todas', el modelo de regresión evaluará el comportamiento global de las probetas a 50 mm/min juntas."
    )
    
    if mezcla_elegida == 'Todas':
        df_est = df_tab4.copy()
    else:
        df_est = df_tab4[df_tab4['Mezcla'] == mezcla_elegida].copy()
        
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
            
    with col_4B:
        if len(df_clean) > 2:
            fig_p, ax_p = plt.subplots(figsize=(9, 5))
            
            sns.scatterplot(data=df_clean, x=pend_selec, y='Rigidez_20', hue='Todas', palette='tab10', s=90, alpha=0.8, ax=ax_p)
            sns.regplot(data=df_clean, x=pend_selec, y='Rigidez_20', scatter=False, color='black', line_kws={'linestyle': '--', 'alpha':0.6}, ax=ax_p)
            
            if 'Probeta' in df_clean.columns:
                for idx, row in df_clean.iterrows():
                    if pd.notna(row['Probeta']):
                        ax_p.annotate(str(row['Probeta']), (row[pend_selec], row['Rigidez_20']),
                                      textcoords="offset points", xytext=(6, 6), ha='left', fontsize=8, alpha=0.8)
            
            ax_p.set_ylabel('Rigidez Real (MPa)')
            ax_p.grid(True, linestyle='--', alpha=0.5)
            
            ax_p.legend(title='Tipo de Probeta', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title_fontsize=9)
            fig_p.tight_layout()
            st.pyplot(fig_p)

    st.markdown("---")
    
    # CARGA Y TENSIÓN VS RIGIDEZ
    st.subheader("4.2. Parámetros de Rotura vs Rigidez")
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
                        ax_c.annotate(str(row['Probeta']), (row['Carga_pico'], row['Rigidez_20']),
                                      textcoords="offset points", xytext=(5,5), ha='left', fontsize=8, alpha=0.8)
            
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
                            ax_t.annotate(str(row['Probeta']), (row['Tension_Rotura'], row['Rigidez_20']),
                                          textcoords="offset points", xytext=(5,5), ha='left', fontsize=8, alpha=0.8)
                
                ax_t.set_xlabel('Tensión Rotura (MPa)')
                ax_t.set_ylabel('Rigidez (MPa)')
                ax_t.grid(True, alpha=0.3)
                ax_t.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
                fig_t.tight_layout()
                
                st.pyplot(fig_t)
                st.caption(f"R²: {mod_t.rsquared:.4f} | p-value: {mod_t.pvalues['Tension_Rotura']:.4f}")
        else:
            st.info("No se dispone de Tensión de Rotura para este conjunto.")

    st.markdown("---")
    
    # 4. MAPAS DE CALOR DIVIDIDOS
    st.subheader("4.3. Matrices de Correlación")
    
    st.info("""
    **¿Qué significan estos números?**
    
    Los valores mostrados representan el **Coeficiente de Correlación de Pearson ($r$)**. 
    Este coeficiente evalúa estadísticamente si existe una relación lineal entre dos variables:
    """)
    
    macro_params = ['Rigidez_20', 'Carga_pico']
    if 'Tension_Rotura' in df_est.columns:
        macro_params.append('Tension_Rotura')
        
    if len(df_est) > 2 and pendientes_validas:
        # A. MAPA DE CALOR RECTANGULAR: MACRO VS MACRO Y PENDIENTES
        st.markdown("**A. Correlación cruzada: Parámetros Macro (Rigidez y Rotura) vs Pendientes pre-pico (y entre sí)**")
        corr_full = df_est[macro_params + pendientes_validas].corr()
        
        corr_macro_pendientes = corr_full.loc[macro_params, macro_params + pendientes_validas]
        
        fig_heat1, ax_heat1 = plt.subplots(figsize=(20, len(macro_params) * 1.5))
        sns.heatmap(corr_macro_pendientes, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=0.5, ax=ax_heat1, annot_kws={"size": 9})
        ax_heat1.tick_params(axis='y', rotation=0)
        st.pyplot(fig_heat1)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # B. MAPA DE CALOR CUADRADO: PENDIENTES VS PENDIENTES (TIPO TRIÁNGULO INFERIOR)
        st.markdown("**B. Correlación interna: Análisis de colinealidad entre las distintas franjas de Pendiente**")
        st.markdown("*Nota: Solo se muestra el triángulo inferior de la matriz geométrica para facilitar la lectura, ya que la matriz es simétrica. Se han añadido 4 decimales para distinguir micro-variaciones entre pendientes altamente colineales.*")
        
        corr_pendientes = df_est[pendientes_validas].corr()
        
        # Creamos una máscara para ocultar la diagonal superior
        mask = np.triu(np.ones_like(corr_pendientes, dtype=bool))
        
        fig_heat2, ax_heat2 = plt.subplots(figsize=(16, 12))
        # Usamos fmt=".4f" para añadir más decimales y la mask para recortarlo
        sns.heatmap(corr_pendientes, mask=mask, annot=True, cmap='YlGnBu', fmt=".4f", linewidths=0.5, ax=ax_heat2, annot_kws={"size": 8})
        st.pyplot(fig_heat2)
    else:
        st.warning("No hay suficientes datos o faltan columnas de pendientes para generar los mapas de calor.")

# --- PESTAÑA 5: REPARTO ENERGÉTICO ---
with tab5:
    st.header("5. Balance Energético de Fractura")
    st.markdown("Análisis de la energía para iniciar la fisuración (**Gf prepico**) vs resistencia residual (**Gf postpico**).")
    
    if 'Gf_prepico' in df.columns and 'Gf_postpico' in df.columns:
        
        col5A, col5B, col5C = st.columns(3)
        
        with col5A:
            mezclas_disp = df['Mezcla'].dropna().unique().tolist()
            filtro_mezcla_t5 = st.selectbox("Elegir Mezcla:", ["Todas"] + mezclas_disp, key="t5_mezcla")
            
        with col5B:
            eje_x_energia = st.selectbox("Agrupar gráficas por (Eje X):", ["Todas", "Envejecimiento", "RAP", "Velocidad"], key="t5_var")
            
        with col5C:
            tipo_vista = st.radio(
                "Modo de representación:", 
                ["Valores Absolutos (J/m²)", "Porcentaje Relativo (100%)"], 
                horizontal=False,
                key="energia_modo"
            )
            
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
            
            fig_stack = px.bar(
                df_melted, x=eje_x_energia, y='Valor', color='Fase_Energia', barmode='stack', text_auto='.1f',
                color_discrete_map={'Gf_prepico': '#3498db', 'Gf_postpico': '#e67e22'}
            )
            fig_stack.update_layout(height=550, xaxis_tickangle=-45, yaxis_title=eje_y_titulo)
            st.plotly_chart(fig_stack, use_container_width=True)
        else:
            st.warning("No hay datos disponibles para la mezcla seleccionada.")
    else:
        st.error("No se han encontrado las columnas 'Gf_prepico' y 'Gf_postpico'.")