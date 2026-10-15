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
from scipy.integrate import trapezoid
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
    * **m75 / m_xx:** fragilidad, asociada a la pendiente postpico y propagación de fisura.
    * **Carga Pico:** resistencia asociada al inicio de la formación de la fisura.
    * **CT_Index:** índice global de tolerancia a la fisuración.
    * **Centroide Pre-pico:** Momento estático real integrado de la curva inicial.
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
                    df[col] = [val.strip() if isinstance(val, str) else val for val in df[col]]
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

@st.cache_data
def agregar_centroide_prepico(df_main):
    """Calcula el centroide real integrando datos_brutos.xlsx y lo une al dataframe principal."""
    if not os.path.exists('datos_brutos.xlsx'):
        return df_main
    try:
        df_b = pd.read_excel('datos_brutos.xlsx', sheet_name=0, header=None)
        resultados = []
        num_columnas = df_b.shape[1]
        
        for i in range(0, num_columnas, 2):
            mezcla_orig = str(df_b.iloc[0, i]).strip()
            if mezcla_orig.lower() == 'nan': continue
            
            start_row = 4
            for row_idx in range(4, 10):
                try:
                    float(df_b.iloc[row_idx, i])
                    start_row = row_idx
                    break
                except:
                    pass
            
            x_data = pd.to_numeric(df_b.iloc[start_row:, i], errors='coerce').dropna().values
            y_data = pd.to_numeric(df_b.iloc[start_row:, i+1], errors='coerce').dropna().values
            
            min_len = min(len(x_data), len(y_data))
            if min_len == 0: continue
            
            x = x_data[:min_len]
            y = y_data[:min_len]
            
            idx_peak = np.argmax(y)
            if idx_peak == 0: continue 
            
            x_peak = x[idx_peak] 
            y_peak = y[idx_peak] 
            
            x_pre = x[:idx_peak+1]
            y_pre = y[:idx_peak+1]
            
            # Integración del centroide real pre-pico
            Qx_real = 0.5 * trapezoid(y_pre**2, x_pre)
            area_real = trapezoid(y_pre, x_pre)
            
            centroide_y = Qx_real / area_real if area_real > 0 else np.nan
            
            resultados.append({
                'Mezcla_Orig': mezcla_orig,
                'y_peak': y_peak,
                'Centroide_Prepico': centroide_y
            })
            
        df_calc = pd.DataFrame(resultados)
        if df_calc.empty: return df_main
        
        df_calc['Mezcla_Merge'] = df_calc['Mezcla_Orig'].astype(str).str.strip().str.upper().str.replace(' ', '').str.replace('BBTM11B', 'BBTM11')
        
        df_main_merge = df_main.copy()
        df_main_merge['_orig_index'] = df_main_merge.index
        df_main_merge['Mezcla_Merge'] = df_main_merge['Mezcla'].astype(str).str.strip().str.upper().str.replace(' ', '').str.replace('BBTM11B', 'BBTM11')
        
        df_merged = pd.merge(df_main_merge, df_calc, on=['Mezcla_Merge'], how='left')
        df_merged['diff_pico'] = abs(df_merged['y_peak'] - df_merged['Carga_pico'])
        df_merged = df_merged.sort_values('diff_pico')
        
        # Quedarnos con el "match" más exacto para cada fila original
        df_merged = df_merged.drop_duplicates(subset=['_orig_index'])
        
        # Eliminar las asignaciones que superen la tolerancia (curva distinta)
        df_merged.loc[df_merged['diff_pico'] >= 1.0, 'Centroide_Prepico'] = np.nan
        
        # Restaurar orden original y limpiar columnas temporales
        df_merged = df_merged.sort_values('_orig_index').drop(columns=['Mezcla_Merge', 'Mezcla_Orig', 'y_peak', 'diff_pico', '_orig_index']).reset_index(drop=True)
        
        return df_merged
    except Exception as e:
        st.warning(f"Error procesando centroides: {e}")
        return df_main

# Cargar y enriquecer los datos
df = cargar_datos()
df = agregar_centroide_prepico(df)

if 'l75' in df.columns and 'm75' in df.columns:
    df['abs_m75'] = df['m75'].abs()
    df['Ratio_Flexibilidad'] = (df['l75'] / df['abs_m75'].replace(0, np.nan)) * 1e6

# 1. Índice de fisuración clásico (Basado en Tensión de Rotura)
if 'Gf_prepico' in df.columns and 'Tension_Rotura' in df.columns and 'abs_m75' in df.columns:
    df['Tension_Rotura_SI'] = df['Tension_Rotura'] * 1e6
    df['Num_Fisuracion'] = df['Gf_prepico'] * (df['Tension_Rotura_SI'] / 3)
    df['Indice_Fisuracion'] = df['Num_Fisuracion'] / df['abs_m75'].replace(0, np.nan)

# 2. Índice de fisuración propuesto (Basado en Tensión Equivalente del Centroide Pre-pico sin /3)
if 'Gf_prepico' in df.columns and 'Centroide_Prepico' in df.columns and 'abs_m75' in df.columns and 'Espesor' in df.columns and 'Diametro' in df.columns:
    # Convertir la fuerza del centroide (kN) a Tensión equivalente (MPa)
    df['Tension_Centroide'] = (df['Centroide_Prepico'] * 1000) / (df['Espesor'] * df['Diametro'])
    # Convertir a Pascales (SI)
    df['Tension_Centroide_SI'] = df['Tension_Centroide'] * 1e6
    
    # Eliminado el factor / 3
    df['Num_Fisuracion_Centroide'] = df['Gf_prepico'] * df['Tension_Centroide_SI']
    df['Indice_Fisuracion_Centroide'] = df['Num_Fisuracion_Centroide'] / df['abs_m75'].replace(0, np.nan)

# 3. Variables combinadas para análisis de correlación (Apartado 1.5)
if 'Gf_prepico' in df.columns:
    if 'Centroide_Prepico' in df.columns:
        df['Gf_x_Centroide'] = df['Gf_prepico'] * df['Centroide_Prepico']
    if 'Tension_Rotura' in df.columns:
        df['Gf_x_TR'] = df['Gf_prepico'] * df['Tension_Rotura']

st.markdown("---")

# ---------------------------------------------------------
# 2. PESTAÑAS DE NAVEGACIÓN
# ---------------------------------------------------------
tab_parametrico, tab_stiff, tab_fisuracion, tab_velocidades = st.tabs([
    "Análisis Paramétrico", 
    "Diagrama Stiff",
    "Índice de fisuración",
    "Efecto Velocidad (50 vs 2)"
])

# --- PESTAÑA 1: ANÁLISIS PARAMÉTRICO ---
with tab_parametrico:
    st.header("1. Análisis Paramétrico Integrado")
    
    opciones_var = ['Carga_pico', 'CT_Index', 'Gf', 'Gf_prepico', 'Gf_postpico', 'm75', 'l75', 'Ratio_Flexibilidad', 'lpeak']
    if 'Tension_Rotura' in df.columns: opciones_var.append('Tension_Rotura')
    if 'Centroide_Prepico' in df.columns: opciones_var.append('Centroide_Prepico')
    
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
    
    modo_barra_t1 = 'relative' if eje_x_bar == color_bar or eje_x_bar == 'Todas' else 'group'
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
                x_ctrl = df_ctrl['Todas'].iloc[0] if eje_x_bar == 'Todas' else df_ctrl[eje_x_bar].iloc[0]
                x_e2 = df_e2['Todas'].iloc[0] if eje_x_bar == 'Todas' else df_e2[eje_x_bar].iloc[0]
                
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
    if 'Gf_prepico' in df.columns and 'Gf_postpico' in df.columns:
        col3A, col3B, col3C = st.columns(3)
        with col3A:
            filtro_mezcla_t5 = st.selectbox("Elegir Mezcla:", ["Todas"] + mezclas_disp_t3, key="t5_mezcla")
        with col3B:
            eje_x_energia = st.selectbox("Agrupar gráficas por (Eje X):", ["Todas", "Envejecimiento", "RAP", "Velocidad"], key="t5_var")
        with col3C:
            tipo_vista = st.radio("Modo de representación:", ["Valores Absolutos (J/m²)", "Porcentaje Relativo (100%)"], horizontal=False, key="energia_modo")
            
        df_energia = df[df['Mezcla'] == filtro_mezcla_t5].copy() if filtro_mezcla_t5 != "Todas" else df.copy()
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
    
    features_obj = ['Carga_pico', 'Gf_prepico', 'Gf_postpico', 'abs_m75', 'l75', 'lpeak', 'Tension_Rotura', 'Centroide_Prepico']
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

    st.markdown("---")

    st.subheader("1.5. Correlación cruzada: Parámetros Macro vs Pendientes pre-pico")
    
    # Hemos añadido Gf_prepico y las multiplicaciones cruzadas al eje Y para estudiar su relación
    macro_cols = ['Rigidez_20', 'Carga_pico', 'Tension_Rotura', 'Centroide_Prepico', 'Gf_prepico', 'Gf_x_Centroide', 'Gf_x_TR']
    macro_cols_presentes = [c for c in macro_cols if c in df.columns]
    
    slope_cols = [c for c in df.columns if c.startswith('m') and (c.endswith('_10') or c.endswith('_20'))]
    cols_correlacion = macro_cols_presentes + slope_cols
    
    if len(cols_correlacion) > len(macro_cols_presentes):
        df_corr = df[cols_correlacion].dropna()
        if not df_corr.empty:
            matriz_corr = df_corr.corr()
            corr_subset = matriz_corr.loc[macro_cols_presentes, cols_correlacion]
            
            fig_heatmap = px.imshow(
                corr_subset,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale='YlGnBu',
                range_color=[0.65, 1.0],
                labels=dict(color="Correlación")
            )
            
            fig_heatmap.update_traces(xgap=1, ygap=1)
            fig_heatmap.update_layout(
                height=350 + (len(macro_cols_presentes)*20), # Ajuste dinámico de altura
                xaxis_tickangle=-45,
                margin=dict(t=30, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("No hay suficientes datos numéricos concurrentes para calcular las correlaciones.")
    else:
        st.warning("No se encontraron las columnas de pendientes pre-pico en los datos cargados.")

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
            
        st.markdown("---")
        st.subheader("2.1. Justificación Estadística del Diagrama Stiff")
        
        val_params = ['Carga_pico', 'l75', 'Gf_prepico', 'Gf_postpico', 'Rigidez_20', 'm75', 'Ratio_Flexibilidad']
        missing_val_cols = [c for c in val_params if c not in df.columns]
        
        if not missing_val_cols:
            df_val = df[df['Velocidad'] == 50][val_params].dropna()
            
            if len(df_val) > 5:
                col4A_st, col4B_st = st.columns(2)
                with col4A_st:
                    corr_matrix = df_val.corr()
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu', center=0, vmin=-1, vmax=1, fmt=".2f", linewidths=0.5, ax=ax_corr)
                    st.pyplot(fig_corr)
                    
                with col4B_st:
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

# --- PESTAÑA 3: ÍNDICE DE FISURACIÓN ---
with tab_fisuracion:
    st.header("3. Espacio de Diseño: Tolerancia a la Fisuración y Durabilidad")
    st.markdown("Comparativa entre el índice clásico apoyado en la Tensión de Rotura frente a la aproximación integral empleando la Tensión Equivalente del Centroide Pre-pico.")
    
    def render_indice_fisuracion(col_idx, lbl_idx, id_suffix, sec_num_1, sec_num_2, threshold_x=None):
        df_tab_lab = df[df['Velocidad'] == 50].copy()
        if col_idx not in df_tab_lab.columns:
            st.warning(f"No se pudo calcular el {lbl_idx}. Faltan datos necesarios para su formulación.")
            return
            
        st.subheader(f"{sec_num_1}. Comparativa del Índice por Mezcla")
        
        col_labA, col_labB = st.columns(2)
        with col_labA:
            agrup_x_lab = st.selectbox("Eje X:", ['Todas', 'Mezcla', 'Envejecimiento', 'RAP'], key=f"lab_x_fis_{id_suffix}", index=0)
        with col_labB:
            color_lab = st.selectbox("Color:", ['Todas', 'Envejecimiento', 'RAP', 'Mezcla'], key=f"lab_color_fis_{id_suffix}", index=0)
            
        agrupacion_lab = [agrup_x_lab] if agrup_x_lab == color_lab else [agrup_x_lab, color_lab]
        df_barras_lab = df_tab_lab.dropna(subset=[col_idx]).groupby(agrupacion_lab)[col_idx].mean().reset_index()
        df_barras_lab[color_lab] = df_barras_lab[color_lab].astype(str)
        
        modo_barra_lab = 'relative' if agrup_x_lab == color_lab or agrup_x_lab == 'Todas' else 'group'
        
        fig_bar_lab = px.bar(
            df_barras_lab, x=agrup_x_lab, y=col_idx, color=color_lab, 
            barmode=modo_barra_lab, text_auto='.2f', color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        for m in df_tab_lab['Mezcla'].dropna().unique():
            for r in df_tab_lab['RAP'].dropna().unique():
                df_ctrl = df_tab_lab[(df_tab_lab['Mezcla'] == m) & (df_tab_lab['RAP'] == r) & (df_tab_lab['Envejecimiento'] == 'Control')]
                df_e2 = df_tab_lab[(df_tab_lab['Mezcla'] == m) & (df_tab_lab['RAP'] == r) & (df_tab_lab['Envejecimiento'] == 'E2')]
                
                if not df_ctrl.empty and not df_e2.empty:
                    if agrup_x_lab == 'Todas':
                        x_ctrl = df_ctrl['Todas'].iloc[0]
                        x_e2 = df_e2['Todas'].iloc[0]
                    else:
                        x_ctrl = df_ctrl[agrup_x_lab].iloc[0]
                        x_e2 = df_e2[agrup_x_lab].iloc[0]
                    
                    if x_ctrl != x_e2:
                        y_ctrl = df_ctrl[col_idx].mean()
                        y_e2 = df_e2[col_idx].mean()
                        
                        fig_bar_lab.add_trace(go.Scatter(
                            x=[x_ctrl, x_e2],
                            y=[y_ctrl, y_e2],
                            mode='lines+markers',
                            line=dict(width=2.5, dash='dot', color='rgba(50,50,50,0.8)'),
                            marker=dict(size=7, color='black', symbol='circle'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

        fig_bar_lab.update_layout(height=450, xaxis_tickangle=-45, yaxis_title=lbl_idx)
        st.plotly_chart(fig_bar_lab, use_container_width=True)
        
        st.markdown("---")
        st.subheader(f"{sec_num_2}. Análisis de Durabilidad: Retención de Fisuración")
        
        df_dur = df_tab_lab.groupby(['Mezcla', 'RAP', 'Envejecimiento'])[col_idx].mean().reset_index()
        try:
            df_pivot = df_dur.pivot(index=['Mezcla', 'RAP'], columns='Envejecimiento', values=col_idx).reset_index()
            if 'Control' in df_pivot.columns and 'E2' in df_pivot.columns:
                df_pivot['Indice_Durabilidad_E2 (%)'] = (df_pivot['E2'] / df_pivot['Control']) * 100
                col_durA, col_durB = st.columns([1.5, 1])
                
                with col_durA:
                    fig_dur = px.scatter(
                        df_pivot, x='Control', y='Indice_Durabilidad_E2 (%)', 
                        color='Mezcla', symbol='RAP', hover_data=['Mezcla', 'RAP', 'E2'], 
                        labels={'Control': f'{lbl_idx} (Control)', 'Indice_Durabilidad_E2 (%)': 'Índice de Durabilidad E2 (%)'}, 
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )
                    fig_dur.update_traces(marker=dict(size=14, line=dict(width=1, color='DarkSlateGrey')))
                    
                    if threshold_x is None:
                        threshold_x = df_pivot['Control'].median()
                        if pd.isna(threshold_x) or threshold_x == 0:
                            threshold_x = df_pivot['Control'].max() / 2
                            
                    max_x_dur = df_pivot['Control'].max() * 1.15
                    if max_x_dur < threshold_x * 1.5:
                        max_x_dur = threshold_x * 1.5
                        
                    max_y_dur = max(df_pivot['Indice_Durabilidad_E2 (%)'].max() * 1.15, 120) if not df_pivot.empty else 120
                    
                    fig_dur.add_shape(type="line", x0=threshold_x, y0=0, x1=threshold_x, y1=max_y_dur, line=dict(color="black", dash="dash", width=2), layer="below")
                    fig_dur.add_shape(type="line", x0=0, y0=80, x1=max_x_dur, y1=80, line=dict(color="red", dash="dash", width=2), layer="below")
                    fig_dur.add_shape(type="line", x0=threshold_x, y0=30, x1=max_x_dur, y1=30, line=dict(color="red", dash="dash", width=2), layer="below")
                    
                    arrow_style = dict(showarrow=False, font=dict(size=40, color="rgba(200, 50, 50, 0.15)"))
                    arrow_style_green = dict(showarrow=False, font=dict(size=40, color="rgba(50, 200, 50, 0.15)"))
                    
                    mid_x_left = threshold_x / 2
                    mid_x_right = threshold_x + (max_x_dur - threshold_x) / 2
                    
                    fig_dur.add_annotation(x=mid_x_left, y=80 + (max_y_dur - 80)/2, text="⬆⬆", **arrow_style_green)
                    fig_dur.add_annotation(x=mid_x_left, y=40, text="⬇⬇", **arrow_style)
                    fig_dur.add_annotation(x=mid_x_right, y=80 + (max_y_dur - 80)/2, text="⬆⬆", **arrow_style_green)
                    fig_dur.add_annotation(x=mid_x_right, y=30 + (80 - 30)/2, text="⬆", **arrow_style_green)
                    fig_dur.add_annotation(x=mid_x_right, y=15, text="⬇⬇", **arrow_style)
                    
                    fig_dur.update_layout(
                        height=550, 
                        title=f"Espacio de Diseño de Durabilidad - {lbl_idx}",
                        xaxis=dict(range=[0, max_x_dur], rangemode='tozero'),
                        yaxis=dict(range=[0, max_y_dur], rangemode='tozero')
                    )
                    st.plotly_chart(fig_dur, use_container_width=True)
                with col_durB:
                    st.markdown("**Tabla de Diagnóstico**")
                    cols_show = ['Mezcla', 'RAP', 'Control', 'E2', 'Indice_Durabilidad_E2 (%)']
                    st.dataframe(df_pivot[cols_show].style.format({'Control': '{:.2f}', 'E2': '{:.2f}', 'Indice_Durabilidad_E2 (%)': '{:.1f}%'}, na_rep="-"))
        except Exception as e:
            st.warning(f"No se pudo generar el espacio de durabilidad: {e}")

    # --- Renderizado de los dos bloques ---
    st.markdown("### A. Formulación original (Aproximación por Tensión de Rotura)")
    render_indice_fisuracion('Indice_Fisuracion', 'Índice de Fisuración (TR/3)', 'TR', '3.1', '3.2', threshold_x=500)
    
    st.markdown("---")
    
    st.markdown("### B. Nueva formulación (Integración exacta por Centroide Pre-pico)")
    # En la nueva formulación permitimos que calcule la mediana como umbral X dinámico
    render_indice_fisuracion('Indice_Fisuracion_Centroide', 'Índice de Fisuración (Tensión Equivalente Centroide)', 'CENT', '3.3', '3.4', threshold_x=None)

    st.markdown("---")
    st.markdown("### C. Comparativa Directa de Índices")
    st.subheader("3.5. Correlación: IF Clásico vs IF Propuesto")
    
    df_comp_idx = df[df['Velocidad'] == 50].dropna(subset=['Indice_Fisuracion', 'Indice_Fisuracion_Centroide']).copy()
    
    if not df_comp_idx.empty:
        col_comp1, col_comp2 = st.columns([2, 1])
        
        with col_comp1:
            fig_comp = px.scatter(
                df_comp_idx, x='Indice_Fisuracion', y='Indice_Fisuracion_Centroide',
                color='Mezcla', symbol='Envejecimiento', hover_data=['RAP', 'Probeta'],
                labels={
                    'Indice_Fisuracion': 'IF Clásico (Tensión Rotura / 3)',
                    'Indice_Fisuracion_Centroide': 'IF Propuesto (Tensión Centroide)'
                },
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig_comp.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
            
            if len(df_comp_idx) > 1:
                X_comp = sm.add_constant(df_comp_idx['Indice_Fisuracion'])
                model_comp = sm.OLS(df_comp_idx['Indice_Fisuracion_Centroide'], X_comp).fit()
                r2_comp = model_comp.rsquared
                
                x_rango_comp = np.array([df_comp_idx['Indice_Fisuracion'].min(), df_comp_idx['Indice_Fisuracion'].max()])
                if len(model_comp.params) > 1:
                    y_rango_comp = model_comp.params.iloc[0] + model_comp.params.iloc[1] * x_rango_comp
                else:
                    y_rango_comp = model_comp.params.iloc[0] * x_rango_comp
                
                fig_comp.add_trace(go.Scatter(
                    x=x_rango_comp, y=y_rango_comp, mode='lines',
                    line=dict(color='black', width=2, dash='dash'),
                    name=f'Ajuste Lineal (R² = {r2_comp:.3f})',
                    showlegend=True, hoverinfo='skip'
                ))
            
            fig_comp.update_layout(height=500)
            st.plotly_chart(fig_comp, use_container_width=True)
            
        with col_comp2:
            st.info("💡 **Análisis de Correlación**\n\nEste gráfico evalúa si la nueva formulación basada en la integración exacta del centroide ordena y penaliza las mezclas de forma equivalente al índice clásico.")
            
            if len(df_comp_idx) > 1:
                st.markdown(f"**Coeficiente de Determinación ($R^2$):** `{r2_comp:.3f}`")
                st.markdown("""
                *Un $R^2$ alto indica que ambas métricas varían en la misma proporción matemática.*
                """)
                
                spearman_corr = df_comp_idx['Indice_Fisuracion'].corr(df_comp_idx['Indice_Fisuracion_Centroide'], method='spearman')
                st.markdown(f"**Correlación de Spearman:** `{spearman_corr:.3f}`")
                st.markdown("""
                *Indica en qué medida ambas métricas mantienen el mismo orden de "ranking" (de mejor a peor mezcla).*
                """)
                
        st.markdown("---")
        st.subheader("3.6. Análisis de Desviación (Clásico vs Propuesto)")
        
        # Calcular el error/desviación relativo asumiendo como "real/base" la formulación integral
        df_comp_idx['Desviacion_%'] = (abs(df_comp_idx['Indice_Fisuracion'] - df_comp_idx['Indice_Fisuracion_Centroide']) / df_comp_idx['Indice_Fisuracion_Centroide'].replace(0, np.nan)) * 100

        col_err1, col_err2 = st.columns([2, 1])
        with col_err1:
            fig_err = px.box(
                df_comp_idx, x='Mezcla', y='Desviacion_%', color='Mezcla',
                points='all', hover_data=['RAP', 'Probeta', 'Envejecimiento'],
                labels={'Desviacion_%': 'Desviación Relativa (%)'},
                title='Distribución de la Desviación por Mezcla',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_err.update_layout(height=400)
            st.plotly_chart(fig_err, use_container_width=True)
            
        with col_err2:
            st.markdown(f"### Desviación Mediana Global: **{df_comp_idx['Desviacion_%'].median():.2f}%**")
            st.info("Calculada tomando como referencia matemática la formulación del Centroide (considerada más exacta al integrar toda el área pre-pico). El porcentaje indica cuánto yerra la asimilación clásica del triángulo.")
            
            df_err_stats = df_comp_idx.groupby('Mezcla').agg({'Desviacion_%': 'median'}).reset_index().rename(columns={'Desviacion_%': 'Desviación Mediana (%)'}).sort_values('Desviación Mediana (%)')
            st.markdown("**Desviación por Mezcla:**")
            st.dataframe(df_err_stats.style.background_gradient(cmap='Reds', subset=['Desviación Mediana (%)']).format({'Desviación Mediana (%)': '{:.2f}%'}))
            
    else:
        st.warning("No hay suficientes datos para comparar ambos índices.")

# --- PESTAÑA 4: COMPARATIVA DE VELOCIDADES ---
with tab_velocidades:
    st.header("4. Análisis del Efecto de la Velocidad de Ensayo (50 vs 2 mm/min)")
    st.markdown("Esta sección evalúa si el comportamiento relativo de las mezclas se mantiene proporcional al reducir la velocidad de 50 a 2 mm/min.")
    
    if 'Indice_Fisuracion' in df.columns and 2 in df['Velocidad'].values and 50 in df['Velocidad'].values:
        df_mean_vel = df.groupby(['Mezcla', 'RAP', 'Envejecimiento', 'Velocidad'])['Indice_Fisuracion'].mean().reset_index()
        df_pivot_vel = df_mean_vel.pivot(index=['Mezcla', 'RAP', 'Envejecimiento'], columns='Velocidad', values='Indice_Fisuracion').reset_index()
        df_pivot_vel.columns.name = None
        df_pivot_vel = df_pivot_vel.rename(columns={2: 'IF_2', 50: 'IF_50'})
        df_pivot_vel['Grupo'] = df_pivot_vel['Mezcla'] + " (" + df_pivot_vel['RAP'] + ")"
        
        df_valid_vel = df_pivot_vel.dropna(subset=['IF_2', 'IF_50'])
        
        st.markdown("---")
        st.subheader("4.1. Correlación del Índice de Fisuración (Magnitud)")
        col6A, col6B = st.columns([2, 1])
        
        with col6A:
            fig_vel_if = px.scatter(
                df_valid_vel, x='IF_2', y='IF_50', 
                color='Grupo', symbol='Envejecimiento',
                hover_data=['Mezcla', 'RAP', 'Envejecimiento'],
                labels={'IF_2': 'Índice a 2 mm/min', 'IF_50': 'Índice a 50 mm/min'},
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig_vel_if.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
            
            X_val = df_valid_vel['IF_2'].values
            y_val = df_valid_vel['IF_50'].values
            if len(X_val) > 1:
                model_no_int = sm.OLS(y_val, X_val).fit()
                pendiente = model_no_int.params[0]
                
                x_line = np.array([0, X_val.max() * 1.1])
                y_line = pendiente * x_line
                
                fig_vel_if.add_trace(go.Scatter(
                    x=x_line, y=y_line, mode='lines', 
                    line=dict(color='black', width=2, dash='dash'), 
                    name=f"Tendencia Global (y = {pendiente:.2f}x)"
                ))
            
            fig_vel_if.update_layout(height=500)
            st.plotly_chart(fig_vel_if, use_container_width=True)
            
        with col6B:
            st.info("💡 **Análisis de Proporcionalidad**\n\nSi los puntos se alinean a lo largo de una recta que pasa por el origen, significa que cambiar la velocidad actúa como un **escalar** sobre el índice.")
            st.dataframe(df_valid_vel[['Grupo', 'Envejecimiento', 'IF_2', 'IF_50']].style.format({'IF_2': '{:.1f}', 'IF_50': '{:.1f}'}))
            
        st.markdown("---")
        st.subheader("4.2. Comparativa del Índice de Durabilidad (Retención %)")
        
        df_dur_v = df_mean_vel.pivot(index=['Mezcla', 'RAP', 'Velocidad'], columns='Envejecimiento', values='Indice_Fisuracion').reset_index()
        
        if 'Control' in df_dur_v.columns and 'E2' in df_dur_v.columns:
            df_dur_v['Durabilidad'] = (df_dur_v['E2'] / df_dur_v['Control']) * 100
            
            df_dur_pivot = df_dur_v.pivot(index=['Mezcla', 'RAP'], columns='Velocidad', values='Durabilidad').reset_index()
            df_dur_pivot.columns.name = None
            df_dur_pivot = df_dur_pivot.rename(columns={2: 'Dur_2', 50: 'Dur_50'})
            df_dur_pivot['Grupo'] = df_dur_pivot['Mezcla'] + " (" + df_dur_pivot['RAP'] + ")"
            
            df_valid_dur = df_dur_pivot.dropna(subset=['Dur_2', 'Dur_50'])
            
            if not df_valid_dur.empty:
                col6C, col6D = st.columns([2, 1])
                with col6C:
                    fig_vel_dur = px.scatter(
                        df_valid_dur, x='Dur_2', y='Dur_50', 
                        color='Grupo', hover_data=['Mezcla', 'RAP'],
                        labels={'Dur_2': 'Durabilidad (%) a 2 mm/min', 'Dur_50': 'Durabilidad (%) a 50 mm/min'},
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )
                    fig_vel_dur.update_traces(marker=dict(size=14, opacity=0.9, line=dict(width=1, color='DarkSlateGrey')))
                    
                    max_dur = max(df_valid_dur['Dur_2'].max(), df_valid_dur['Dur_50'].max()) * 1.15
                    min_dur = min(df_valid_dur['Dur_2'].min(), df_valid_dur['Dur_50'].min()) * 0.85
                    
                    fig_vel_dur.add_shape(type="line", x0=min_dur, y0=min_dur, x1=max_dur, y1=max_dur, line=dict(color="rgba(150,150,150,0.5)", dash="dash", width=2), layer="below")
                    fig_vel_dur.add_trace(go.Scatter(x=[min_dur, max_dur], y=[min_dur, max_dur], mode='lines', line=dict(color='rgba(0,0,0,0)'), name='Línea de Igualdad (y=x)', showlegend=True))

                    fig_vel_dur.update_layout(height=500, xaxis=dict(range=[min_dur, max_dur]), yaxis=dict(range=[min_dur, max_dur]))
                    st.plotly_chart(fig_vel_dur, use_container_width=True)
                    
                with col6D:
                    st.info("💡 **Consistencia de la Durabilidad**\n\nSi la velocidad no altera la susceptibilidad relativa al envejecimiento, los puntos deberían estar en la línea **y=x**.")
                    st.dataframe(df_valid_dur[['Grupo', 'Dur_2', 'Dur_50']].style.format({'Dur_2': '{:.1f}%', 'Dur_50': '{:.1f}%'}))
            else:
                st.warning("No hay suficientes datos cruzados de Control y E2 para comparar la durabilidad a ambas velocidades.")
        else:
            st.warning("Faltan estados de envejecimiento (Control o E2) para calcular el Índice de Durabilidad.")
            
        st.markdown("---")
        st.subheader("4.3. Evolución del Índice de Fisuración por Velocidad")
        
        vel_sel = st.radio("Velocidad de ensayo a visualizar (mm/min):", [50, 2], horizontal=True, key="radio_vel_tendencia")
        df_tend = df[df['Velocidad'] == vel_sel].dropna(subset=['Indice_Fisuracion']).copy()
        
        if not df_tend.empty:
            df_barras_tend = df_tend.groupby(['Todas', 'Mezcla'])['Indice_Fisuracion'].mean().reset_index()
            
            fig_tend = px.bar(
                df_barras_tend, x='Todas', y='Indice_Fisuracion', color='Mezcla',
                text_auto='.1f', color_discrete_sequence=px.colors.qualitative.Set2
            )
            
            for m in df_tend['Mezcla'].unique():
                for r in df_tend['RAP'].unique():
                    df_c = df_tend[(df_tend['Mezcla'] == m) & (df_tend['RAP'] == r) & (df_tend['Envejecimiento'] == 'Control')]
                    df_e2 = df_tend[(df_tend['Mezcla'] == m) & (df_tend['RAP'] == r) & (df_tend['Envejecimiento'] == 'E2')]
                    
                    if not df_c.empty and not df_e2.empty:
                        x_c = df_c['Todas'].iloc[0]
                        x_e2 = df_e2['Todas'].iloc[0]
                        y_c = df_c['Indice_Fisuracion'].mean()
                        y_e2 = df_e2['Indice_Fisuracion'].mean()
                        
                        fig_tend.add_trace(go.Scatter(
                            x=[x_c, x_e2], y=[y_c, y_e2], mode='lines+markers',
                            line=dict(width=2.5, dash='dot', color='rgba(50,50,50,0.8)'),
                            marker=dict(size=8, color='black'), showlegend=False, hoverinfo='skip'
                        ))
            
            fig_tend.update_layout(height=500, xaxis_tickangle=-45, yaxis_title="Índice de Fisuración", xaxis_title="Todas")
            st.plotly_chart(fig_tend, use_container_width=True)
        else:
            st.warning(f"No hay datos de Índice de Fisuración para {vel_sel} mm/min.")
            
    else:
        st.warning("No se encontraron ensayos simultáneos a 2 mm/min y a 50 mm/min en la base de datos para realizar la comparativa.")