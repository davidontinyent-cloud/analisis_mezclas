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
from scipy.stats import f_oneway
import os

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
    if not os.path.exists('datos.xlsx'):
        st.error("No se encontró el archivo 'datos.xlsx'. Asegúrate de que el archivo Excel esté en el mismo directorio.")
        st.stop()
        
    try:
        df = pd.read_excel('datos.xlsx', engine='openpyxl')
        df.columns = df.columns.astype(str).str.strip()
        df = df.replace('-', np.nan)
        
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    cleaned_list = [val.strip() if isinstance(val, str) else val for val in df[col]]
                    df[col] = cleaned_list
                except Exception:
                    pass
        
        cols_numericas = ['Carga_pico', 'CT_Index', 'Gf', 'Gf_prepico', 'Gf_postpico', 'm75', 'l75', 'lpeak', 'Rigidez_20']
        for col in cols_numericas:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
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
        
    except Exception as e:
        st.error(f"Hubo un error cargando el archivo de datos: {e}")
        st.stop()
    return None

df = cargar_datos()

if 'l75' in df.columns and 'm75' in df.columns:
    df['abs_m75'] = df['m75'].abs()
    df['Ratio_Flexibilidad'] = (df['l75'] / df['abs_m75'].replace(0, np.nan)) * 1e6

if 'Gf_prepico' in df.columns and 'Tension_Rotura' in df.columns and 'abs_m75' in df.columns:
    df['Tension_Rotura_SI'] = df['Tension_Rotura'] * 1e6
    df['Num_Fisuracion'] = df['Gf_prepico'] * (df['Tension_Rotura_SI'] / 3)
    df['Indice_Fisuracion'] = df['Num_Fisuracion'] / df['abs_m75'].replace(0, np.nan)
    df['TR_sobre_m75'] = df['Tension_Rotura_SI'] / df['abs_m75'].replace(0, np.nan)

st.markdown("---")

# ---------------------------------------------------------
# 2. PESTAÑAS DE NAVEGACIÓN
# ---------------------------------------------------------
tab_parametrico, tab_stiff, tab_reg, tab_rigidez_tension, tab_fisuracion, tab_brutos = st.tabs([
    "Análisis Paramétrico", 
    "Diagrama Stiff",
    "Comparativa Rigidez",
    "Rigidez vs Tensión",
    "Índice de fisuración",
    "Curvas Fuerza-Recorrido"
])

# --- PESTAÑA 1: ANÁLISIS PARAMÉTRICO ---
with tab_parametrico:
    st.header("1. Análisis Paramétrico Integrado")
    
    opciones_var = ['Carga_pico', 'CT_Index', 'Gf', 'Gf_prepico', 'Gf_postpico', 'm75', 'l75', 'Ratio_Flexibilidad', 'lpeak']
    if 'Tension_Rotura' in df.columns: opciones_var.append('Tension_Rotura')
    
    st.subheader("1.1. Comparativa Directa por Parámetro (Medias)")
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
    
    if eje_x_bar == color_bar or eje_x_bar == 'Todas':
        modo_barra_t1 = 'relative'
    else:
        modo_barra_t1 = 'group'
        
    fmt_texto_t2 = '.4f' if param_bar == 'lpeak' else '.2f'
    
    fig_bar = px.bar(
        df_barras, x=eje_x_bar, y=param_bar, color=color_bar, 
        barmode=modo_barra_t1, text_auto=fmt_texto_t2, color_discrete_sequence=px.colors.qualitative.Alphabet if color_bar == 'Todas' else px.colors.qualitative.Set2
    )
    
    for m in df['Mezcla'].dropna().unique():
        for r in df['RAP'].dropna().unique():
            df_ctrl = df[(df['Mezcla'] == m) & (df['RAP'] == r) & (df['Envejecimiento'] == 'Control')]
            df_e2 = df[(df['Mezcla'] == m) & (df['RAP'] == r) & (df['Envejecimiento'] == 'E2')]
            
            if not df_ctrl.empty and not df_e2.empty:
                x_ctrl = df_ctrl[eje_x_bar].iloc[0]
                x_e2 = df_e2[eje_x_bar].iloc[0]
                
                if x_ctrl != x_e2:
                    y_ctrl = df_ctrl[param_bar].mean()
                    y_e2 = df_e2[param_bar].mean()
                    
                    fig_bar.add_trace(go.Scatter(
                        x=[x_ctrl, x_e2],
                        y=[y_ctrl, y_e2],
                        mode='lines+markers',
                        line=dict(width=2.5, dash='dot', color='rgba(50,50,50,0.8)'),
                        marker=dict(size=7, color='black', symbol='circle'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

    fig_bar.update_layout(height=450, xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    
    st.subheader("1.2. Análisis de Dispersión y Variabilidad")
    st.markdown("Las cajas muestran la dispersión real de los datos, la mediana central y los posibles valores atípicos (*outliers*).")
    
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

    fig_box.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    st.subheader("1.3. Balance Energético de Fractura")
    st.markdown("Evaluación proporcional de la energía consumida antes (Gf_prepico) y después (Gf_postpico) de la carga máxima.")
    
    if 'Gf_prepico' in df.columns and 'Gf_postpico' in df.columns:
        col3A, col3B, col3C = st.columns(3)
        with col3A:
            filtro_mezcla_t5 = st.selectbox("Elegir Mezcla:", ["Todas"] + mezclas_disp_t3, key="t5_mezcla")
        with col3B:
            eje_x_energia = st.selectbox("Agrupar gráficas por (Eje X):", ["Todas", "Envejecimiento", "RAP", "Velocidad"], key="t5_var")
        with col3C:
            tipo_vista = st.radio("Modo de representación:", ["Valores Absolutos (J/m²)", "Porcentaje Relativo (100%)"], horizontal=False, key="energia_modo")
            
        if filtro_mezcla_t5 != "Todas":
            df_energia = df[df['Mezcla'] == filtro_mezcla_t5].copy()
        else:
            df_energia = df.copy()
            
        df_energia = df_energia[~((df_energia['Mezcla'] == 'AC22') & (df_energia['RAP'] != 'Sin_RAP'))]
            
        if not df_energia.empty:
            col_v50, col_v2 = st.columns(2)
            for vel, col_layout in zip([50, 2], [col_v50, col_v2]):
                with col_layout:
                    st.markdown(f"**Velocidad: {vel} mm/min**")
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
                        fig_stack.update_layout(height=400, xaxis_tickangle=-45, yaxis_title=eje_y_titulo)
                        st.plotly_chart(fig_stack, use_container_width=True)
                    else:
                        st.info(f"No hay datos para {vel} mm/min con los filtros actuales.")
        else:
            st.warning("No hay datos disponibles tras aplicar los filtros de exclusión.")

    st.markdown("---")

    st.subheader("1.4. Análisis de Poder Discriminatorio (ANOVA)")
    st.markdown("""
    Este filtro estadístico (ANOVA de una vía) calcula el Valor F (varianza entre grupos frente a dentro de los grupos) aplicando los datos a **50 mm/min**. 
    **A mayor F, mayor poder separador tiene la variable** para identificar correctamente el objetivo.
    """)
    
    col_tgt1, col_tgt2 = st.columns(2)
    with col_tgt1:
        target_col = st.radio("Selecciona la variable objetivo a clasificar:", ["Mezcla", "Envejecimiento"], horizontal=True, key="anova_target")
    
    df_ml = df[df['Velocidad'] == 50].copy()
    
    with col_tgt2:
        if target_col == "Envejecimiento":
            st.info("💡 Sugerencia: Al evaluar el envejecimiento, es mejor filtrar por una mezcla concreta para evitar que las diferencias estructurales enmascaren el efecto térmico.")
            filtro_mezcla_ml = st.selectbox("Filtrar datos por Mezcla:", ["Todas juntas"] + df_ml['Mezcla'].dropna().unique().tolist())
            if filtro_mezcla_ml != "Todas juntas":
                df_ml = df_ml[df_ml['Mezcla'] == filtro_mezcla_ml]
        else:
            st.info("💡 Para clasificar mezclas, el algoritmo necesita verlas todas simultáneamente. Si lo deseas, puedes aislar un estado de envejecimiento concreto.")
            filtro_env_ml = st.selectbox("Filtrar datos por Envejecimiento:", ["Todos juntos"] + df_ml['Envejecimiento'].dropna().unique().tolist())
            if filtro_env_ml != "Todos juntos":
                df_ml = df_ml[df_ml['Envejecimiento'] == filtro_env_ml]
    
    features_obj = ['Carga_pico', 'Gf_prepico', 'Gf_postpico', 'abs_m75', 'l75', 'lpeak', 'Tension_Rotura']
    valid_features = [f for f in features_obj if f in df_ml.columns]
    df_ml = df_ml.dropna(subset=valid_features + [target_col])
    
    if len(df_ml[target_col].unique()) > 1 and len(df_ml) >= 5:
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
    else:
        st.warning("No hay suficientes datos válidos o no hay suficientes categorías distintas para entrenar el modelo en esta selección.")


# --- PESTAÑA 2: DIAGRAMA STIFF ---
with tab_stiff:
    st.header("2. Firma Mecánica: Diagrama de Stiff Diferencial")
    
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
        st.subheader("2.1. Justificación Estadística del Diagrama Stiff")
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


# --- PESTAÑA 3: COMPARATIVA RIGIDEZ ---
with tab_reg:
    st.header("3. Comparativa Rigidez")
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
    
    st.subheader("3.1. Influencia de la Pendiente Pre-Pico")
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
    
    st.subheader("3.2. Parámetros de Rotura vs Rigidez")
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
    st.subheader("3.3. Matrices de Correlación")
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


# --- PESTAÑA 4: RIGIDEZ VS TENSIÓN ---
with tab_rigidez_tension:
    st.header("4. Análisis de Interacción: Rigidez vs Tensión de Rotura")
    st.markdown("""
    Para evitar que la alta varianza estructural de una mezcla AC enmascare el comportamiento viscoelástico de una SMA, 
    la estrategia analítica fundamental es utilizar el tipo de mezcla como filtro primario y aplicar el concepto estadístico de 
    **"Múltiplos Pequeños"** (Small Multiples). Esto previene la Paradoja de Simpson, donde una tendencia global contradice el comportamiento real de los subgrupos.
    """)
    st.markdown("⚠️ *Análisis filtrado únicamente para ensayos a **50 mm/min**.*")
    
    if 'Tension_Rotura' in df.columns and 'Rigidez_20' in df.columns:
        df_rt = df[df['Velocidad'] == 50].dropna(subset=['Tension_Rotura', 'Rigidez_20', 'Mezcla', 'Envejecimiento', 'RAP']).copy()
        
        if not df_rt.empty:
            
            st.markdown("---")
            st.subheader("4.1. Relación Global: Tensión vs Rigidez (Todas las Probetas)")
            st.markdown("Diagrama de dispersión global con **líneas de tendencia independientes por mezcla** y una **Línea de Ajuste Total** (en negro discontinuo) que evalúa el coeficiente de correlación de todas las probetas en conjunto.")
            
            df_global = df_rt.copy()
            
            X_tot = sm.add_constant(df_global['Tension_Rotura'])
            mod_tot = sm.OLS(df_global['Rigidez_20'], X_tot).fit()
            m_tot = mod_tot.params.iloc[1] if len(mod_tot.params) > 1 else 0
            b_tot = mod_tot.params.iloc[0]
            r2_tot = mod_tot.rsquared

            fig_global_rt = px.scatter(
                df_global, x="Tension_Rotura", y="Rigidez_20", color="Mezcla",
                trendline="ols",
                hover_data=['Probeta', 'RAP', 'Envejecimiento'],
                labels={"Tension_Rotura": "Tensión de Rotura (MPa)", "Rigidez_20": "Rigidez a 20ºC (MPa)"},
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            
            x_min_tot, x_max_tot = df_global['Tension_Rotura'].min(), df_global['Tension_Rotura'].max()
            x_rango = np.array([x_min_tot, x_max_tot])
            y_rango = b_tot + m_tot * x_rango
            
            fig_global_rt.add_trace(go.Scatter(
                x=x_rango, y=y_rango,
                mode='lines',
                line=dict(color='black', width=3, dash='dash'),
                name=f'Tendencia TOTAL (R² = {r2_tot:.3f})'
            ))
            
            fig_global_rt.update_traces(marker=dict(size=8, opacity=0.8))
            fig_global_rt.update_layout(height=500)
            st.plotly_chart(fig_global_rt, use_container_width=True)
            
            st.markdown("---")
            st.subheader("4.2. Análisis Aislado: Rigidez y Tensión de Rotura")
            st.markdown("""
            Panel independiente para cada mezcla cuantificando el impacto del envejecimiento y del RAP. 
            Permite ver si el RAP actúa puramente como rigidizador o aporta fuerza, y si la varianza (ancho de caja) se vuelve inestable en E2.
            """)
            
            var_aislada = st.radio("Selecciona la variable a diseccionar:", ["Tension_Rotura", "Rigidez_20"], horizontal=True)
            y_label_aislada = "Tensión de Rotura (MPa)" if var_aislada == "Tension_Rotura" else "Rigidez a 20ºC (MPa)"
            
            fig_box_rt = px.box(
                df_rt, x="Envejecimiento", y=var_aislada, color="RAP", facet_col="Mezcla",
                category_orders={"Envejecimiento": ["Control", "E1", "E2"]},
                labels={var_aislada: y_label_aislada},
                color_discrete_sequence=px.colors.qualitative.Set1,
                points="all"
            )
            
            df_medians_rt = df_rt.groupby(['Mezcla', 'Envejecimiento', 'RAP'])[var_aislada].median().reset_index()
            order_env = {'Control': 0, 'E1': 1, 'E2': 2}
            df_medians_rt['orden'] = df_medians_rt['Envejecimiento'].map(order_env)
            df_medians_rt = df_medians_rt.sort_values(['Mezcla', 'RAP', 'orden'])

            for i, mezcla in enumerate(df_rt['Mezcla'].unique()):
                df_mezcla = df_medians_rt[df_medians_rt['Mezcla'] == mezcla]
                for rap in df_mezcla['RAP'].unique():
                    df_rap = df_mezcla[df_mezcla['RAP'] == rap]
                    fig_box_rt.add_trace(go.Scatter(
                        x=df_rap['Envejecimiento'], y=df_rap[var_aislada],
                        mode='lines+markers',
                        line=dict(width=2, dash='dot', color='rgba(50,50,50,0.6)'),
                        marker=dict(size=6, color='rgba(50,50,50,0.8)', symbol='diamond'),
                        showlegend=False,
                        hoverinfo='skip',
                        xaxis=f'x{i+1}' if i > 0 else 'x',
                        yaxis=f'y{i+1}' if i > 0 else 'y'
                    ))

            fig_box_rt.update_yaxes(matches=None) 
            fig_box_rt.update_layout(height=500, boxmode="group")
            st.plotly_chart(fig_box_rt, use_container_width=True)
            
            st.markdown("---")
            st.subheader("4.3. Relación Bivariante: Trayectorias Cinemáticas por Mezcla")
            st.markdown("""
            Representación de Rigidez (X) frente a Tensión de Rotura (Y). Las líneas conectan los **centroides** de los datos 
            (`Control → E1 → E2`) trazando el vector de deterioro **separado por contenido de RAP**.
            *   **Vector a 45º:** La mezcla gana fuerza al mismo ritmo que se endurece.
            *   **Vector plano hacia la derecha:** La mezcla se rigidiza sin ganar resistencia (fragilización inminente).
            """)
            
            mezclas_rt = df_rt['Mezcla'].unique()
            cols_rt = st.columns(len(mezclas_rt))
            
            for i, mezcla in enumerate(mezclas_rt):
                with cols_rt[i]:
                    st.markdown(f"**Mezcla: {mezcla}**")
                    df_m = df_rt[df_rt['Mezcla'] == mezcla]
                    
                    rap_types = df_m['RAP'].dropna().unique()
                    palette = px.colors.qualitative.Dark2
                    color_map = {rap: palette[j % len(palette)] for j, rap in enumerate(rap_types)}
                    
                    fig_scatter_rt = px.scatter(
                        df_m, x="Rigidez_20", y="Tension_Rotura", color="RAP",
                        hover_data=['Probeta', 'Envejecimiento'],
                        labels={"Rigidez_20": "Rigidez (MPa)", "Tension_Rotura": "Tensión Rotura"},
                        color_discrete_map=color_map
                    )
                    
                    centroides = df_m.groupby(['RAP', 'Envejecimiento'])[['Rigidez_20', 'Tension_Rotura']].mean().reset_index()
                    orden_env = {'Control': 0, 'E1': 1, 'E2': 2}
                    centroides['orden'] = centroides['Envejecimiento'].map(orden_env)
                    
                    for rap_val in rap_types:
                        cent_subset = centroides[centroides['RAP'] == rap_val].sort_values('orden').dropna()
                        
                        if len(cent_subset) > 1:
                            fig_scatter_rt.add_trace(go.Scatter(
                                x=cent_subset['Rigidez_20'], y=cent_subset['Tension_Rotura'],
                                mode='lines+markers+text',
                                marker=dict(symbol='star', size=14, color=color_map[rap_val], line=dict(width=1, color='black')),
                                line=dict(color=color_map[rap_val], width=2, dash='dot'),
                                text=cent_subset['Envejecimiento'],
                                textposition="top center",
                                showlegend=False,
                                name=f"Trayectoria ({rap_val})"
                            ))
                    
                    fig_scatter_rt.update_layout(height=400, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
                    if i > 0:
                        fig_scatter_rt.update_yaxes(title_text="")
                        
                    st.plotly_chart(fig_scatter_rt, use_container_width=True)
            
            st.caption("Puntos y trayectorias coloreadas por RAP. Las estrellas y líneas punteadas representan la migración del centroide de cada subgrupo.")

            st.markdown("---")
            st.subheader("4.4. Cuantificación Lineal (Coeficientes de Pearson)")
            st.markdown("Cálculo del coeficiente de correlación ($r$) entre Rigidez y Tensión aislando cada clúster de Mezcla y RAP.")
            
            res_corr = []
            for name, group in df_rt.groupby(['Mezcla', 'RAP']):
                if len(group) > 2:
                    corr = group['Rigidez_20'].corr(group['Tension_Rotura'])
                    res_corr.append({'Mezcla': name[0], 'RAP': name[1], 'Correlación de Pearson (r)': corr})
            
            if res_corr:
                df_corr = pd.DataFrame(res_corr)
                st.dataframe(df_corr.style.format({'Correlación de Pearson (r)': '{:.3f}'})
                                         .background_gradient(cmap='RdYlGn', subset=['Correlación de Pearson (r)'], vmin=-1, vmax=1))
            else:
                st.info("No hay suficientes datos por clúster para calcular correlaciones individuales.")
                
        else:
            st.warning("No hay suficientes datos válidos para este análisis.")
    else:
        st.error("Para poder utilizar esta pestaña es necesario que el archivo contenga las columnas 'Tension_Rotura' y 'Rigidez_20'.")


# --- PESTAÑA 5: ÍNDICE DE FISURACIÓN ---
with tab_fisuracion:
    st.header("5. Índice de Fisuración (Experimental)")
    st.markdown("Espacio de pruebas paramétrico. Evalúa la relación directa entre la energía inicial, la capacidad de carga máxima y la penalización por la fragilidad post-pico.")
    st.markdown("⚠️ *Análisis filtrado únicamente para ensayos a **50 mm/min**.*")
    
    if 'Indice_Fisuracion' in df.columns:
        df_tab_lab = df[df['Velocidad'] == 50].copy()
        
        col_idx = 'Indice_Fisuracion'
        col_num = 'Num_Fisuracion'
        col_den = 'abs_m75'
        col_energia = 'Gf_prepico'
        lbl_energia = 'Energía Pre-pico (J/m²)'
        lbl_num = '(Tensión / 3) (Pa) × Gf_prepico (J/m²)'
        lbl_idx = 'Índice de Fisuración'
        lbl_den = '|m75| (N/m)'
        fmt_texto = '.1f'
            
        st.markdown("---")
        
        st.subheader("5.1. Comparativa del Índice por Mezcla")
        
        col_labA, col_labB = st.columns(2)
        with col_labA:
            agrup_x_lab = st.selectbox("Eje X:", ['Todas', 'Mezcla', 'Envejecimiento', 'RAP'], key="lab_x_fis", index=0)
        with col_labB:
            color_lab = st.selectbox("Color:", ['Todas', 'Envejecimiento', 'RAP', 'Mezcla'], key="lab_color_fis", index=0)
            
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
        
        st.subheader("5.4. Relación: Tensión de Rotura vs Energía Pre-pico")
        st.markdown("Gráfico que enfrenta exclusivamente la capacidad de carga (Tensión de Rotura) frente a la energía de fractura inicial (Gf_prepico), base del numerador de la fórmula.")
        
        df_valid_lab5 = df_tab_lab.dropna(subset=['Tension_Rotura', col_energia]).copy()
        
        df_valid_lab5['Grupo_Analisis'] = df_valid_lab5.apply(
            lambda row: f"AC22 ({row['RAP']})" if row['Mezcla'] == 'AC22' else row['Mezcla'], axis=1
        )
        
        grupos_unicos = df_valid_lab5['Grupo_Analisis'].dropna().unique()
        colores_plotly = px.colors.qualitative.Set1
        mapa_colores = {g: colores_plotly[i % len(colores_plotly)] for i, g in enumerate(grupos_unicos)}

        fig_scatter_lab5 = px.scatter(
            df_valid_lab5, 
            x=col_energia, y='Tension_Rotura', 
            color='Grupo_Analisis', 
            symbol='Envejecimiento',
            hover_data=['Probeta', 'RAP', 'Mezcla'],
            labels={col_energia: lbl_energia, 'Tension_Rotura': 'Tensión de Rotura (MPa)', 'Grupo_Analisis': 'Mezcla'},
            color_discrete_map=mapa_colores
        )
        
        fig_scatter_lab5.update_traces(marker=dict(size=9, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
        
        for grupo in grupos_unicos:
            df_sub = df_valid_lab5[df_valid_lab5['Grupo_Analisis'] == grupo]
            if len(df_sub) > 1:
                X_sm = sm.add_constant(df_sub[col_energia])
                modelo = sm.OLS(df_sub['Tension_Rotura'], X_sm).fit()
                
                x_min, x_max = df_sub[col_energia].min(), df_sub[col_energia].max()
                x_rango = np.array([x_min, x_max])
                
                if len(modelo.params) > 1:
                    y_rango = modelo.params.iloc[0] + modelo.params.iloc[1] * x_rango
                else:
                    y_rango = modelo.params.iloc[0] * x_rango
                    
                fig_scatter_lab5.add_trace(go.Scatter(
                    x=x_rango, y=y_rango,
                    mode='lines',
                    line=dict(color=mapa_colores[grupo], width=2.5, dash='dot'),
                    name=f"Tendencia {grupo}",
                    showlegend=True,
                    hoverinfo='skip'
                ))

        fig_scatter_lab5.update_layout(height=480)
        st.plotly_chart(fig_scatter_lab5, use_container_width=True)

        st.markdown("---")
        
        col_labE, col_labF = st.columns(2)
        
        with col_labE:
            st.subheader(f"5.5. Tensión / Fragilidad vs {col_energia}")
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
            st.subheader(f"5.6. Tensión vs Fragilidad (Burbuja = {col_energia})")
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
        
        st.header("5.7. Análisis de Durabilidad: Retención de Fisuración")
        st.markdown("""
        Plantear un diagrama de espacio de diseño entre el estado inicial y la tasa de retención al envejecer es una herramienta fundamental en metodologías como el Diseño Equilibrado (BMD).

        * **Eje X:** Índice de Fisuración Inicial ($IF_{Control}$). Representa la calidad base tras la compactación.
        * **Eje Y:** Tasa de Retención ($IF_{E2} / IF_{Control} \cdot 100$). Representa qué porcentaje de sus propiedades sobrevive a la vida útil.
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
                    
                    x_threshold = 500
                    y_top = 80
                    y_bot = 30
                    
                    max_x_dur = max(df_pivot['Control'].max() * 1.15, x_threshold * 1.5) if not df_pivot.empty else 800
                    max_y_dur = max(df_pivot['Tasa_Retencion_E2 (%)'].max() * 1.15, 120) if not df_pivot.empty else 120
                    
                    fig_dur.add_shape(type="line", x0=x_threshold, y0=0, x1=x_threshold, y1=max_y_dur, 
                                      line=dict(color="black", dash="dash", width=2), layer="below")
                    
                    fig_dur.add_shape(type="line", x0=0, y0=y_top, x1=max_x_dur, y1=y_top, 
                                      line=dict(color="red", dash="dash", width=2), layer="below")
                                      
                    fig_dur.add_shape(type="line", x0=x_threshold, y0=y_bot, x1=max_x_dur, y1=y_bot, 
                                      line=dict(color="red", dash="dash", width=2), layer="below")
                                      
                    fig_dur.add_shape(type="line", x0=0, y0=0, x1=x_threshold, y1=y_top, 
                                      line=dict(color="gray", dash="dot", width=1.5), layer="below")
                    
                    arrow_style = dict(showarrow=False, font=dict(size=36, color="rgba(200, 50, 50, 0.3)"))
                    arrow_style_green = dict(showarrow=False, font=dict(size=36, color="rgba(50, 200, 50, 0.3)"))
                    
                    mid_x_left = x_threshold / 2
                    mid_x_right = x_threshold + (max_x_dur - x_threshold) / 2
                    
                    fig_dur.add_annotation(x=mid_x_left, y=y_top - (y_top/4), text="⬇", **arrow_style)
                    fig_dur.add_annotation(x=mid_x_left, y=(y_top/4), text="⬇⬇", **arrow_style)
                    fig_dur.add_annotation(x=mid_x_right, y=y_top + (max_y_dur - y_top)/2, text="⬆⬆", **arrow_style_green)
                    fig_dur.add_annotation(x=mid_x_right, y=y_bot + (y_top - y_bot)/2, text="⬆", **arrow_style_green)
                    fig_dur.add_annotation(x=mid_x_right, y=y_bot / 2, text="⬇⬇", **arrow_style)
                    
                    fig_dur.update_layout(
                        height=550, 
                        title="Espacio de Diseño de Durabilidad",
                        xaxis=dict(range=[0, max_x_dur]),
                        yaxis=dict(range=[0, max_y_dur])
                    )
                        
                    st.plotly_chart(fig_dur, use_container_width=True)
                    
                with col_durB:
                    st.markdown("**Diagnóstico de Durabilidad de la Industria**")
                    st.markdown("""
                    **Tasa de Retención (Aging Ratio):**
                    Es la métrica más estandarizada en la industria para medir el daño de la mezcla a largo plazo. 
                    Se define como el porcentaje del índice de fisuración que sobrevive a los protocolos de envejecimiento severo en horno. 
                    """)
                    
                    st.markdown("**Tabla de Diagnóstico**")
                    df_show = df_pivot.copy()
                    
                    cols_show = ['Mezcla', 'RAP', 'Control']
                    if 'E1' in df_show.columns: cols_show.append('E1')
                    cols_show.append('E2')
                    cols_show.append('Tasa_Retencion_E2 (%)')
                    
                    df_show = df_show[cols_show]
                    
                    format_dict = {'Control': '{:.1f}', 'E2': '{:.1f}', 'Tasa_Retencion_E2 (%)': '{:.1f}%'}
                    if 'E1' in df_show.columns:
                        format_dict['E1'] = '{:.1f}'
                    
                    st.dataframe(df_show.style.format(format_dict, na_rep="-"))
                    
            else:
                st.warning("Faltan datos en estado 'Control' o 'E2' en las series para poder trazar el ciclo completo de durabilidad.")
        except Exception as e:
            st.warning(f"No se ha podido computar el análisis de durabilidad debido a la estructura actual de los datos: {e}")
            
    else:
        st.error("No se puede calcular el nuevo índice. Faltan datos necesarios en el archivo.")


# --- PESTAÑA 6: CURVAS FUERZA-RECORRIDO ---
with tab_brutos:
    st.header("6. Curvas Características Fuerza - Recorrido y Distancia Relativa de Centroides")
    st.markdown("""
    Selecciona **dos mezclas** simultáneamente para superponer sus curvas representativas de Control (E0, Sin_RAP), 
    visualizar sus triángulos pre-pico y calcular automáticamente la distancia relativa entre sus baricentros mecánicos.
    """)
    
    try:
        df_raw_brutos = pd.read_excel('datos_brutos.xlsx', header=None)
        curvas_representativas = {}
        
        for i in range(0, df_raw_brutos.shape[1]-1, 2):
            mezcla = str(df_raw_brutos.iloc[0, i]).strip()
            rap = str(df_raw_brutos.iloc[1, i]).strip()
            env = str(df_raw_brutos.iloc[2, i]).strip()
            
            if rap == 'Sin_RAP' and env == 'E0':
                sub = df_raw_brutos.iloc[4:, [i, i+1]].dropna()
                if not sub.empty:
                    sub.columns = ['Recorrido', 'Fuerza']
                    sub['Recorrido'] = pd.to_numeric(sub['Recorrido'], errors='coerce')
                    sub['Fuerza'] = pd.to_numeric(sub['Fuerza'], errors='coerce')
                    sub = sub.dropna().sort_values('Recorrido')
                    
                    if mezcla not in curvas_representativas:
                        curvas_representativas[mezcla] = []
                    curvas_representativas[mezcla].append(sub)
        
        mejores_curvas = {}
        for m, lista in curvas_representativas.items():
            picos = [p['Fuerza'].max() for p in lista]
            mediana_pico = np.median(picos)
            idx_cercano = np.argmin([abs(p - mediana_pico) for p in picos])
            mejores_curvas[m] = lista[idx_cercano]
        
        if mejores_curvas:
            lista_mezclas = list(mejores_curvas.keys())
            
            default_sel = lista_mezclas[:2] if len(lista_mezclas) >= 2 else lista_mezclas
            
            seleccion_dual = st.multiselect(
                "Selecciona exactamente 2 mezclas para comparar:", 
                options=lista_mezclas, 
                default=default_sel,
                max_selections=2,
                key="multiselect_mezclas_dual"
            )
            
            fig_dual = go.Figure()
            colores_base = {'AC16': '#2ca02c', 'AC22': '#d62728', 'BBTM': '#1f77b4', 'SMA': '#9467bd'}
            
            for m, curva in mejores_curvas.items():
                if m not in seleccion_dual:
                    fig_dual.add_trace(go.Scatter(
                        x=curva['Recorrido'], y=curva['Fuerza'],
                        mode='lines',
                        name=f"Fondo: {m}",
                        line=dict(width=1.5, color='gray'),
                        opacity=0.2,
                        hoverinfo='skip'
                    ))
            
            centroides_calculados = {}
            
            for m in seleccion_dual:
                curva_sel = mejores_curvas[m]
                color_sel = colores_base.get(m, '#000000')
                
                fig_dual.add_trace(go.Scatter(
                    x=curva_sel['Recorrido'], y=curva_sel['Fuerza'],
                    mode='lines',
                    name=f"Curva: {m}",
                    line=dict(width=3.5, color=color_sel),
                    opacity=1.0
                ))
                
                idx_pico = curva_sel['Fuerza'].idxmax()
                x_pico = curva_sel.loc[idx_pico, 'Recorrido']
                y_pico = curva_sel.loc[idx_pico, 'Fuerza']
                
                curva_pre = curva_sel[curva_sel['Recorrido'] <= x_pico]
                x_ini = curva_pre['Recorrido'].iloc[0]
                y_ini = curva_pre['Fuerza'].iloc[0]
                
                xA, yA = x_ini, y_ini
                xB, yB = x_pico, y_pico
                xC, yC = x_pico, y_ini
                
                cx = (xA + xB + xC) / 3
                cy = (yA + yB + yC) / 3
                centroides_calculados[m] = (cx, cy)
                
                fig_dual.add_trace(go.Scatter(
                    x=[xA, xB, xC, xA], y=[yA, yB, yC, yA],
                    fill='toself', fillcolor=f'rgba({int(color_sel[1:3], 16)}, {int(color_sel[3:5], 16)}, {int(color_sel[5:7], 16)}, 0.2)',
                    mode='lines', line=dict(color=color_sel, width=2, dash='dash'),
                    name=f'Triángulo {m}'
                ))
                
                fig_dual.add_trace(go.Scatter(
                    x=[cx], y=[cy],
                    mode='markers+text',
                    marker=dict(size=14, color=color_sel, symbol='star'),
                    text=[f"C_{m} ({cx:.2f}, {cy:.2f})"],
                    textposition="top right",
                    name=f'Centroide {m}'
                ))
            
            if len(seleccion_dual) == 2:
                m1, m2 = seleccion_dual[0], seleccion_dual[1]
                c1x, c1y = centroides_calculados[m1]
                c2x, c2y = centroides_calculados[m2]
                
                fig_dual.add_trace(go.Scatter(
                    x=[c1x, c2x], y=[c1y, c2y],
                    mode='lines',
                    line=dict(color='black', width=2, dash='dot'),
                    name='Vector Distancia Relativa',
                    hoverinfo='skip'
                ))
            
            fig_dual.update_layout(
                height=520,
                xaxis_title="Recorrido / Desplazamiento (mm)",
                yaxis_title="Fuerza (kN)",
                title="Superposición de Curvas, Triángulos Pre-pico y Distancia de Centroides",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_dual, use_container_width=True)
            
            # --- NUEVA SECCIÓN DE JUSTIFICACIÓN MATEMÁTICA Y FÍSICA ---
            st.markdown("---")
            with st.expander("📚 Leer Justificación Física y Geométrica del Momento Estático", expanded=True):
                st.markdown(r"""
                
El numerador del Índice de Fractura propuesto (TR/3 * Gf_prepico) se deduce geométricamente al asimilar la curva de la fase pre-pico a un triángulo, donde la energía absorbida equivale al área Gf_prepico = (TR * l_peak) / 2, lo que transforma la ecuación en la expresión (TR^2 * l_peak) / 6. 

Matemáticamente, este término corresponde exactamente a tres veces el Momento Estático de Área (Qx) respecto al eje horizontal de deformación, una magnitud que define la distribución de un área ponderando su distancia a un eje y que se calcula multiplicando el área total por la altura de su centroide (y = TR / 3, dando como resultado Qx = (TR^2 * l_peak) / 6). 

Físicamente, fundamentar el numerador en este "momento energético" dota a la carga de un brazo de palanca vertical, lo que significa que el índice no premia a las mezclas defectuosas que consiguen absorber energía deformándose infinitamente como un material blando; en su lugar, recompensa de forma exponencial a aquellas mezclas que alcanzan su tenacidad soportando altos niveles de tensión estructural (centroide alto), garantizando un esqueleto mineral resistente al ahuellamiento antes de penalizar su fragilidad post-pico mediante el denominador |m75|.

                """)
                
        else:
            st.warning("No se encontraron curvas válidas en el archivo para el estado Control Sin_RAP.")
            
    except Exception as e:
        st.error(f"Error procesando el archivo de datos brutos: {e}")