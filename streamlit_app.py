import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from math import pi
import warnings
warnings.filterwarnings('ignore')

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="⚽ Wyscout Dashboard Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .stRadio > label {
        font-size: 18px;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- FUNCIONES AUXILIARES --------------------
@st.cache_data
def load_data(file_path="Wyscout_League_Export.csv"):
    """Carga los datos con manejo de errores y optimización"""
    try:
        # Intentar diferentes encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                # Limpiar nombres de columnas
                df.columns = df.columns.str.strip()
                return df
            except UnicodeDecodeError:
                continue
        st.error("No se pudo decodificar el archivo CSV")
        return pd.DataFrame()
    except FileNotFoundError:
        st.error(f"❌ Archivo no encontrado: {file_path}")
        st.info("Por favor, asegúrate de que el archivo 'Wyscout_League_Export.csv' está en la carpeta correcta")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        return pd.DataFrame()

def clean_numeric_column(series):
    """Limpia y convierte columnas a numéricas"""
    if series.dtype == 'object':
        # Remover caracteres no numéricos comunes
        series = series.astype(str).str.replace(',', '.')
        series = series.str.replace('%', '')
        series = series.str.strip()
    return pd.to_numeric(series, errors='coerce')

def normalize_value(value, min_val, max_val):
    """Normaliza un valor entre 0 y 100"""
    if pd.isna(value) or pd.isna(min_val) or pd.isna(max_val):
        return 50
    if max_val == min_val:
        return 50
    return ((value - min_val) / (max_val - min_val)) * 100

def get_player_percentiles(player_data, position_df, metrics):
    """Calcula percentiles del jugador respecto a su posición"""
    percentiles = {}
    for metric in metrics:
        if metric in position_df.columns:
            clean_col = clean_numeric_column(position_df[metric])
            player_val = clean_numeric_column(pd.Series([player_data.get(metric, 0)]))[0]
            
            if not pd.isna(player_val):
                # Calcular percentil
                percentile = (clean_col < player_val).sum() / len(clean_col) * 100
                percentiles[metric] = percentile
            else:
                percentiles[metric] = 50
        else:
            percentiles[metric] = 50
    return percentiles

# -------------------- PAGE 1: DATA EXPLORER --------------------
def page_data_explorer(df):
    st.title("📊 Explorador de Datos Wyscout")
    st.markdown("---")
    
    # Métricas generales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Jugadores", len(df))
    with col2:
        st.metric("Total Equipos", df["Team"].nunique() if "Team" in df.columns else 0)
    with col3:
        st.metric("Total Ligas", df["League"].nunique() if "League" in df.columns else 0)
    with col4:
        st.metric("Columnas", len(df.columns))
    
    st.markdown("---")
    st.subheader("🔍 Filtros de Datos")
    
    # Filtros mejorados con multiselect
    col1, col2, col3 = st.columns(3)
    
    leagues = df["League"].dropna().unique().tolist() if "League" in df.columns else []
    teams = df["Team"].dropna().unique().tolist() if "Team" in df.columns else []
    players = df["Player"].dropna().unique().tolist() if "Player" in df.columns else []
    
    with col1:
        selected_leagues = st.multiselect(
            "Selecciona Liga(s):",
            sorted(leagues),
            default=[],
            help="Puedes seleccionar múltiples ligas"
        )
    
    with col2:
        # Filtrar equipos según ligas seleccionadas
        if selected_leagues:
            available_teams = df[df["League"].isin(selected_leagues)]["Team"].dropna().unique().tolist()
        else:
            available_teams = teams
        
        selected_teams = st.multiselect(
            "Selecciona Equipo(s):",
            sorted(available_teams),
            default=[],
            help="Puedes seleccionar múltiples equipos"
        )
    
    with col3:
        # Filtrar jugadores según equipos seleccionados
        if selected_teams:
            available_players = df[df["Team"].isin(selected_teams)]["Player"].dropna().unique().tolist()
        elif selected_leagues:
            available_players = df[df["League"].isin(selected_leagues)]["Player"].dropna().unique().tolist()
        else:
            available_players = players
        
        selected_players = st.multiselect(
            "Selecciona Jugador(es):",
            sorted(available_players),
            default=[],
            help="Puedes seleccionar múltiples jugadores"
        )
    
    # Aplicar filtros
    filtered_df = df.copy()
    
    if selected_leagues:
        filtered_df = filtered_df[filtered_df["League"].isin(selected_leagues)]
    if selected_teams:
        filtered_df = filtered_df[filtered_df["Team"].isin(selected_teams)]
    if selected_players:
        filtered_df = filtered_df[filtered_df["Player"].isin(selected_players)]
    
    # Mostrar estadísticas del filtrado
    st.info(f"📊 Mostrando {len(filtered_df)} de {len(df)} registros")
    
    # Opción para mostrar estadísticas resumidas
    if st.checkbox("Mostrar estadísticas resumidas"):
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.subheader("📈 Estadísticas Resumidas")
            st.dataframe(filtered_df[numeric_cols].describe().round(2))
    
    # Mostrar datos filtrados
    st.subheader("📋 Datos Filtrados")
    st.dataframe(filtered_df, width="stretch", height=400)
    
    # Botones de descarga
    col1, col2 = st.columns(2)
    with col1:
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar CSV",
            data=csv,
            file_name='wyscout_filtered.csv',
            mime='text/csv',
        )
    
    with col2:
        # Crear Excel en memoria
        try:
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name='Data')
            buffer.seek(0)
            
            st.download_button(
                label="📥 Descargar Excel",
                data=buffer,
                file_name='wyscout_filtered.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            )
        except ImportError:
            st.info("Instala 'openpyxl' para exportar a Excel: pip install openpyxl")

# -------------------- PAGE 2: PIZZA/RADAR CHART MEJORADO --------------------
def page_pizza_chart(df):
    st.title("🕸️ Radar Chart de Rendimiento (Pizza Chart)")
    st.markdown("Comparación visual de estadísticas normalizadas por posición")
    st.markdown("---")
    
    if "Main Position" not in df.columns:
        st.error("❌ No se encontró la columna 'Main Position' en los datos")
        return
    
    # Selector de posición
    positions = sorted(df["Main Position"].dropna().unique().tolist())
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_position = st.selectbox(
            "📍 Selecciona Posición:",
            positions,
            help="Los percentiles se calculan respecto a jugadores de la misma posición"
        )
    
    # Filtrar por posición
    position_df = df[df["Main Position"] == selected_position].copy()
    
    if position_df.empty:
        st.warning("No hay jugadores en esta posición")
        return
    
    with col2:
        # Permitir seleccionar múltiples jugadores para comparar
        players = sorted(position_df["Player"].dropna().unique().tolist())
        selected_players = st.multiselect(
            "👥 Selecciona Jugador(es) para comparar:",
            players,
            default=[players[0]] if players else [],
            max_selections=4,
            help="Puedes comparar hasta 4 jugadores simultáneamente"
        )
    
    if not selected_players:
        st.info("Selecciona al menos un jugador para visualizar")
        return
    
    # Definir métricas disponibles con nombres más legibles
    metrics_mapping = {
        "Goals": "Goles",
        "xG": "xG (Goles Esperados)",
        "Assists": "Asistencias",
        "xA": "xA (Asistencias Esperadas)",
        "Duels per 90": "Duelos por 90",
        "Duels won, %": "% Duelos Ganados",
        "Defensive duels per 90": "Duelos Def. por 90",
        "Defensive duels won, %": "% Duelos Def. Ganados",
        "Shots per 90": "Disparos por 90",
        "Shots on target, %": "% Disparos a Puerta",
        "Successful defensive actions per 90": "Acciones Def. por 90",
        "PAdj Interceptions": "Intercepciones Ajustadas",
        "Accurate passes, %": "% Pases Precisos",
        "Crosses per 90": "Centros por 90",
        "Dribbles per 90": "Regates por 90",
        "Successful dribbles, %": "% Regates Exitosos"
    }
    
    # Identificar métricas disponibles en el dataset
    available_metrics = []
    for metric_col, metric_name in metrics_mapping.items():
        if metric_col in position_df.columns:
            available_metrics.append((metric_col, metric_name))
    
    if len(available_metrics) < 3:
        st.error("No hay suficientes métricas disponibles para crear el radar chart")
        return
    
    # Selector de métricas a incluir
    st.subheader("⚙️ Configuración del Radar Chart")
    
    selected_metrics = st.multiselect(
        "Selecciona las métricas a visualizar:",
        options=[m[1] for m in available_metrics],
        default=[m[1] for m in available_metrics[:8]],  # Primeras 8 por defecto
        help="Selecciona entre 3 y 12 métricas para el radar chart"
    )
    
    if len(selected_metrics) < 3:
        st.warning("Selecciona al menos 3 métricas para crear el radar chart")
        return
    
    # Invertir el mapeo para obtener las columnas originales
    selected_metric_cols = [m[0] for m in available_metrics if m[1] in selected_metrics]
    
    # Opción de visualización
    viz_type = st.radio(
        "Tipo de visualización:",
        ["Percentiles (0-100)", "Valores Absolutos", "Valores Normalizados"],
        horizontal=True,
        help="Percentiles: Compara con jugadores de la misma posición | Absolutos: Valores reales | Normalizados: Escala 0-100"
    )
    
    # Preparar datos para el radar chart
    st.subheader("📊 Radar Chart Interactivo")
    
    # Crear el gráfico con Plotly (más interactivo que matplotlib)
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1[:len(selected_players)]
    
    for idx, player_name in enumerate(selected_players):
        player_data = position_df[position_df["Player"] == player_name].iloc[0]
        
        values = []
        hover_texts = []
        
        for metric_col in selected_metric_cols:
            if viz_type == "Percentiles (0-100)":
                # Calcular percentil
                percentiles = get_player_percentiles(
                    player_data,
                    position_df,
                    [metric_col]
                )
                value = percentiles[metric_col]
                hover_texts.append(f"{value:.1f} percentil")
            
            elif viz_type == "Valores Absolutos":
                # Valores reales
                value = clean_numeric_column(pd.Series([player_data[metric_col]]))[0]
                if pd.isna(value):
                    value = 0
                hover_texts.append(f"{value:.2f}")
            
            else:  # Valores Normalizados
                # Normalizar entre min y max de la posición
                clean_col = clean_numeric_column(position_df[metric_col])
                min_val = clean_col.min()
                max_val = clean_col.max()
                player_val = clean_numeric_column(pd.Series([player_data[metric_col]]))[0]
                value = normalize_value(player_val, min_val, max_val)
                hover_texts.append(f"{value:.1f}%")
            
            values.append(value)
        
        # Añadir traza al gráfico
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=selected_metrics,
            fill='toself',
            name=player_name,
            hovertemplate='<b>%{theta}</b><br>Valor: %{r:.1f}<br><extra></extra>',
            line=dict(color=colors[idx], width=2),
            marker=dict(size=8)
        ))
    
    # Configurar el layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100] if viz_type != "Valores Absolutos" else None,
                showticklabels=True,
                ticks='outside'
            ),
            angularaxis=dict(
                direction="clockwise",
                rotation=90
            )
        ),
        showlegend=True,
        title={
            'text': f"Comparación de Rendimiento - {selected_position}",
            'x': 0.5,
            'xanchor': 'center'
        },
        height=600,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Mostrar tabla comparativa
    if st.checkbox("📋 Mostrar tabla comparativa"):
        st.subheader("Tabla Comparativa de Valores")
        
        comparison_data = []
        for player_name in selected_players:
            player_data = position_df[position_df["Player"] == player_name].iloc[0]
            row_data = {"Jugador": player_name}
            
            for metric_col in selected_metric_cols:
                metric_name = metrics_mapping.get(metric_col, metric_col)
                
                if viz_type == "Percentiles (0-100)":
                    percentiles = get_player_percentiles(
                        player_data,
                        position_df,
                        [metric_col]
                    )
                    row_data[metric_name] = f"{percentiles[metric_col]:.1f}%"
                else:
                    value = clean_numeric_column(pd.Series([player_data[metric_col]]))[0]
                    if pd.isna(value):
                        row_data[metric_name] = "N/A"
                    else:
                        row_data[metric_name] = f"{value:.2f}"
            
            comparison_data.append(row_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, width="stretch")
    
    # Análisis adicional
    with st.expander("📊 Análisis Estadístico de la Posición"):
        st.subheader(f"Estadísticas de {selected_position}")
        
        # Mostrar estadísticas de la posición
        stats_cols = st.columns(3)
        
        with stats_cols[0]:
            st.metric("Total Jugadores", len(position_df))
        
        with stats_cols[1]:
            st.metric("Equipos Representados", position_df["Team"].nunique() if "Team" in position_df.columns else 0)
        
        with stats_cols[2]:
            st.metric("Ligas Representadas", position_df["League"].nunique() if "League" in position_df.columns else 0)
        
        # Top performers por métrica
        st.subheader("🏆 Top 5 por Métrica")
        
        metric_to_show = st.selectbox(
            "Selecciona métrica para ver top jugadores:",
            selected_metric_cols,
            format_func=lambda x: metrics_mapping.get(x, x)
        )
        
        clean_metric = clean_numeric_column(position_df[metric_to_show])
        position_df["_temp_metric"] = clean_metric
        top_players = position_df.nlargest(5, "_temp_metric")[["Player", "Team", metric_to_show]]
        st.dataframe(top_players, width="stretch", hide_index=True)

# -------------------- PAGE 3: SCATTER PLOTS MEJORADO --------------------
def page_scatter_plots(df):
    st.title("📈 Análisis Comparativo (Scatter Plots)")
    st.markdown("Visualiza relaciones entre diferentes métricas y encuentra patrones")
    st.markdown("---")
    
    # Configuración de filtros
    col1, col2 = st.columns(2)
    
    with col1:
        # Filtro de liga con opción "Todas"
        leagues = ["Todas las Ligas"] + sorted(df["League"].dropna().unique().tolist())
        selected_league = st.selectbox(
            "🏆 Selecciona Liga:",
            leagues,
            help="Filtra los datos por liga específica o muestra todas"
        )
    
    with col2:
        # Filtro de posición con opción "Todas"
        if "Main Position" in df.columns:
            positions = ["Todas las Posiciones"] + sorted(df["Main Position"].dropna().unique().tolist())
            selected_position = st.selectbox(
                "📍 Filtrar por Posición:",
                positions,
                help="Opcional: filtra por posición específica"
            )
        else:
            selected_position = "Todas las Posiciones"
    
    # Aplicar filtros
    filtered_df = df.copy()
    
    if selected_league != "Todas las Ligas":
        filtered_df = filtered_df[filtered_df["League"] == selected_league]
    
    if selected_position != "Todas las Posiciones" and "Main Position" in df.columns:
        filtered_df = filtered_df[filtered_df["Main Position"] == selected_position]
    
    st.info(f"📊 Mostrando {len(filtered_df)} jugadores")
    
    # Configuración avanzada
    with st.expander("⚙️ Configuración Avanzada"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            point_size = st.slider("Tamaño de puntos:", 5, 20, 10)
        
        with col2:
            opacity = st.slider("Transparencia:", 0.3, 1.0, 0.7, 0.1)
        
        with col3:
            color_by = st.selectbox(
                "Colorear por:",
                ["Team", "Main Position", "Ninguno"],
                index=0
            )
    
    # Definir comparaciones predefinidas
    st.subheader("📊 Comparaciones Predefinidas")
    
    # Tabs para diferentes categorías
    tabs = st.tabs(["⚽ Goles", "🎯 Asistencias", "⚔️ Duelos", "🎯 Disparos", "📐 Personalizado"])
    
    scatter_configs = {
        "Goles": [
            ("xG", "Goals", "Goles Esperados vs Goles Reales", 
             "Jugadores por encima de la línea diagonal superan sus xG"),
            ("Shots per 90", "Goals", "Disparos por 90 vs Goles",
             "Eficiencia goleadora: más goles con menos disparos = mayor eficiencia"),
        ],
        "Asistencias": [
            ("xA", "Assists", "Asistencias Esperadas vs Asistencias Reales",
             "Jugadores por encima de la diagonal superan sus xA"),
            ("Accurate passes, %", "Assists", "Precisión de Pases vs Asistencias",
             "Relación entre precisión en el pase y capacidad asistidora"),
        ],
        "Duelos": [
            ("Duels per 90", "Duels won, %", "Volumen vs Efectividad en Duelos",
             "Jugadores arriba-derecha: muchos duelos con alta efectividad"),
            ("Defensive duels per 90", "Defensive duels won, %", 
             "Volumen vs Efectividad en Duelos Defensivos",
             "Jugadores arriba-derecha: muchos duelos defensivos con alta efectividad"),
        ],
        "Disparos": [
            ("Shots per 90", "Shots on target, %", "Volumen vs Precisión de Disparos",
             "Jugadores arriba-derecha: muchos disparos con alta precisión"),
            ("Shots on target, %", "Goals", "Precisión de Disparos vs Goles",
             "Efectividad: más goles con mayor precisión"),
        ],
    }
    
    # Mostrar scatter plots según la pestaña seleccionada
    for i, (tab_name, tab) in enumerate(zip(["Goles", "Asistencias", "Duelos", "Disparos"], tabs[:-1])):
        with tab:
            for x_col, y_col, title, description in scatter_configs[tab_name]:
                if x_col in filtered_df.columns and y_col in filtered_df.columns:
                    # Limpiar datos numéricos
                    filtered_df[x_col] = clean_numeric_column(filtered_df[x_col])
                    filtered_df[y_col] = clean_numeric_column(filtered_df[y_col])
                    
                    # Crear el scatter plot
                    fig = px.scatter(
                        filtered_df.dropna(subset=[x_col, y_col]),
                        x=x_col,
                        y=y_col,
                        hover_name="Player" if "Player" in filtered_df.columns else None,
                        hover_data=["Team"] if all(c in filtered_df.columns for c in ["Team"]) else None,
                        color=color_by if color_by != "Ninguno" and color_by in filtered_df.columns else None,
                        title=title,
                        template="plotly_white",
                        opacity=opacity
                    )
                    
                    # Personalizar el gráfico
                    fig.update_traces(marker=dict(size=point_size))
                    
                    # Añadir línea de tendencia si es relevante
                    if "vs" in title and ("Goals" in title or "Assists" in title):
                        fig.add_trace(go.Scatter(
                            x=[filtered_df[x_col].min(), filtered_df[x_col].max()],
                            y=[filtered_df[x_col].min(), filtered_df[x_col].max()],
                            mode='lines',
                            name='Línea 1:1',
                            line=dict(dash='dash', color='gray'),
                            showlegend=False
                        ))
                    
                    fig.update_layout(
                        height=500,
                        hovermode='closest',
                        xaxis_title=x_col,
                        yaxis_title=y_col
                    )
                    
                    # Mostrar descripción
                    st.info(f"💡 {description}")
                    
                    # Mostrar el gráfico
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Estadísticas de correlación
                    if st.checkbox(f"Ver correlación para {title}", key=f"corr_{x_col}_{y_col}"):
                        valid_data = filtered_df[[x_col, y_col]].dropna()
                        if len(valid_data) > 2:
                            correlation = valid_data[x_col].corr(valid_data[y_col])
                            st.metric(f"Correlación de Pearson", f"{correlation:.3f}")
                            
                            # Interpretación
                            if abs(correlation) < 0.3:
                                interpretation = "Correlación débil o nula"
                            elif abs(correlation) < 0.7:
                                interpretation = "Correlación moderada"
                            else:
                                interpretation = "Correlación fuerte"
                            
                            st.write(f"**Interpretación:** {interpretation}")
                else:
                    st.warning(f"⚠️ No se encontraron las columnas necesarias: {x_col}, {y_col}")
    
    # Pestaña de gráfico personalizado
    with tabs[-1]:
        st.subheader("🎨 Crea tu propio Scatter Plot")
        
        # Seleccionar columnas numéricas disponibles
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Intentar también con columnas que parecen numéricas pero están como string
        potential_numeric = [col for col in filtered_df.columns if any(
            keyword in col.lower() for keyword in ['goals', 'assists', 'shots', 'passes', 'duels', '%', 'per 90', 'xa', 'xg']
        )]
        
        # Combinar y eliminar duplicados
        all_numeric_cols = list(set(numeric_cols + potential_numeric))
        all_numeric_cols = [col for col in all_numeric_cols if col in filtered_df.columns]
        
        if len(all_numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox("Eje X:", all_numeric_cols, key="custom_x")
            
            with col2:
                y_axis = st.selectbox(
                    "Eje Y:", 
                    [col for col in all_numeric_cols if col != x_axis], 
                    key="custom_y"
                )
            
            if x_axis and y_axis:
                # Limpiar datos
                filtered_df[x_axis] = clean_numeric_column(filtered_df[x_axis])
                filtered_df[y_axis] = clean_numeric_column(filtered_df[y_axis])
                
                # Crear gráfico personalizado
                fig = px.scatter(
                    filtered_df.dropna(subset=[x_axis, y_axis]),
                    x=x_axis,
                    y=y_axis,
                    hover_name="Player" if "Player" in filtered_df.columns else None,
                    hover_data=["Team"] if all(c in filtered_df.columns for c in ["Team"]) else None,
                    color=color_by if color_by != "Ninguno" and color_by in filtered_df.columns else None,
                    title=f"{x_axis} vs {y_axis}",
                    template="plotly_white",
                    opacity=opacity,
                    trendline="ols" if st.checkbox("Mostrar línea de tendencia") else None
                )
                
                fig.update_traces(marker=dict(size=point_size))
                fig.update_layout(height=500, hovermode='closest')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar top jugadores
                if st.checkbox("Mostrar top jugadores en ambas métricas"):
                    st.subheader("🏆 Top Jugadores")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Top 5 en {x_axis}:**")
                        top_x = filtered_df.nlargest(5, x_axis)[["Player", "Team", x_axis]]
                        st.dataframe(top_x, hide_index=True)
                    
                    with col2:
                        st.write(f"**Top 5 en {y_axis}:**")
                        top_y = filtered_df.nlargest(5, y_axis)[["Player", "Team", y_axis]]
                        st.dataframe(top_y, hide_index=True)
        else:
            st.warning("No hay suficientes columnas numéricas para crear un scatter plot personalizado")

# -------------------- PÁGINA 4: ANÁLISIS AVANZADO --------------------
def page_advanced_analysis(df):
    st.title("🔬 Análisis Avanzado")
    st.markdown("Herramientas avanzadas de análisis y visualización")
    st.markdown("---")
    
    analysis_type = st.selectbox(
        "Selecciona tipo de análisis:",
        ["Ranking Multi-Métrica", "Comparador de Jugadores"]
    )
    
    if analysis_type == "Ranking Multi-Métrica":
        st.subheader("🏅 Ranking Multi-Métrica")
        st.write("Crea un ranking personalizado combinando múltiples métricas con pesos")
        
        # Seleccionar métricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        potential_metrics = [col for col in df.columns if any(
            keyword in col.lower() for keyword in ['goals', 'assists', 'shots', 'passes', 'duels', '%', 'per 90', 'xa', 'xg']
        )]
        available_metrics = list(set(numeric_cols + potential_metrics))
        available_metrics = [col for col in available_metrics if col in df.columns]
        
        selected_metrics = st.multiselect(
            "Selecciona métricas para el ranking:",
            available_metrics,
            default=available_metrics[:3] if len(available_metrics) >= 3 else available_metrics
        )
        
        if selected_metrics:
            # Configurar pesos
            st.write("Asigna pesos a cada métrica (la suma debe ser 100%):")
            weights = {}
            cols = st.columns(len(selected_metrics))
            
            for i, metric in enumerate(selected_metrics):
                with cols[i]:
                    weight = st.number_input(
                        f"{metric[:20]}...",
                        min_value=0,
                        max_value=100,
                        value=100//len(selected_metrics),
                        step=5,
                        key=f"weight_{metric}"
                    )
                    weights[metric] = weight / 100
            
            if abs(sum(weights.values()) - 1.0) > 0.01:
                st.warning(f"⚠️ Los pesos suman {sum(weights.values())*100:.0f}%. Deben sumar 100%.")
            else:
                # Calcular ranking
                ranking_df = df.copy()
                
                # Normalizar cada métrica
                for metric in selected_metrics:
                    clean_col = clean_numeric_column(ranking_df[metric])
                    min_val = clean_col.min()
                    max_val = clean_col.max()
                    if max_val > min_val:
                        ranking_df[f"{metric}_norm"] = (clean_col - min_val) / (max_val - min_val)
                    else:
                        ranking_df[f"{metric}_norm"] = 0.5
                
                # Calcular score compuesto
                ranking_df["Score"] = 0
                for metric in selected_metrics:
                    ranking_df["Score"] += ranking_df[f"{metric}_norm"] * weights[metric] * 100
                
                # Mostrar top jugadores
                top_n = st.slider("Mostrar top N jugadores:", 5, 50, 20)
                
                display_cols = ["Player", "Team", "Main Position", "Score"] + selected_metrics
                display_cols = [col for col in display_cols if col in ranking_df.columns]
                
                top_players = ranking_df.nlargest(top_n, "Score")[display_cols]
                top_players["Score"] = top_players["Score"].round(1)
                
                st.dataframe(
                    top_players.style.background_gradient(subset=["Score"], cmap="RdYlGn"),
                    width="stretch",
                    hide_index=True
                )
    
    elif analysis_type == "Comparador de Jugadores":
        st.subheader("⚖️ Comparador Directo de Jugadores")
        
        col1, col2 = st.columns(2)
        
        with col1:
            player1 = st.selectbox(
                "Jugador 1:",
                sorted(df["Player"].dropna().unique()),
                key="comp_p1"
            )
        
        with col2:
            player2 = st.selectbox(
                "Jugador 2:",
                sorted(df["Player"].dropna().unique()),
                key="comp_p2"
            )
        
        if player1 and player2:
            p1_data = df[df["Player"] == player1].iloc[0]
            p2_data = df[df["Player"] == player2].iloc[0]
            
            # Información básica
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### {player1}")
                st.write(f"**Equipo:** {p1_data.get('Team', 'N/A')}")
                st.write(f"**Posición:** {p1_data.get('Main Position', 'N/A')}")
            
            with col2:
                st.markdown(f"### {player2}")
                st.write(f"**Equipo:** {p2_data.get('Team', 'N/A')}")
                st.write(f"**Posición:** {p2_data.get('Main Position', 'N/A')}")
            
            # Comparación de métricas
            st.markdown("---")
            st.subheader("📊 Comparación de Métricas")
            
            # Identificar métricas numéricas comunes
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            potential_metrics = [col for col in df.columns if any(
                keyword in col.lower() for keyword in ['goals', 'assists', 'shots', 'passes', 'duels', '%', 'per 90', 'xa', 'xg']
            )]
            
            comparison_metrics = list(set(numeric_cols + potential_metrics))
            comparison_metrics = [col for col in comparison_metrics if col in df.columns]
            
            comparison_data = []
            for metric in comparison_metrics[:15]:  # Limitar a 15 métricas
                p1_val = clean_numeric_column(pd.Series([p1_data[metric]]))[0]
                p2_val = clean_numeric_column(pd.Series([p2_data[metric]]))[0]
                
                if not pd.isna(p1_val) and not pd.isna(p2_val):
                    diff = p2_val - p1_val
                    diff_pct = (diff / p1_val * 100) if p1_val != 0 else 0
                    
                    comparison_data.append({
                        "Métrica": metric,
                        player1: f"{p1_val:.2f}",
                        player2: f"{p2_val:.2f}",
                        "Diferencia": f"{diff:+.2f}",
                        "Diferencia %": f"{diff_pct:+.1f}%"
                    })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                
                # Aplicar formato condicional
                def highlight_better(row):
                    colors = [''] * len(row)
                    try:
                        val1 = float(row[player1])
                        val2 = float(row[player2])
                        if val1 > val2:
                            colors[1] = 'background-color: #90EE90'  # Verde claro
                        elif val2 > val1:
                            colors[2] = 'background-color: #90EE90'  # Verde claro
                    except:
                        pass
                    return colors
                
                st.dataframe(
                    comparison_df.style.apply(highlight_better, axis=1),
                    width="stretch",
                    hide_index=True
                )

# -------------------- MAIN APP --------------------
def main():
    # Sidebar con logo y navegación
    st.sidebar.markdown(
        """
        <div style='text-align: center'>
            <h1>⚽ Wyscout</h1>
            <h3>Dashboard Pro</h3>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown("---")
    
    # Cargar datos
    df = pd.DataFrame()
    
    # Opción para cargar archivo
    data_source = st.sidebar.radio(
        "📂 Fuente de Datos:",
        ["Archivo Local", "Subir CSV"]
    )
    
    if data_source == "Archivo Local":
        file_path = st.sidebar.text_input(
            "Ruta del archivo:",
            value="Wyscout_League_Export.csv"
        )
        if st.sidebar.button("Cargar Datos"):
            df = load_data(file_path)
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Selecciona archivo CSV:",
            type=['csv']
        )
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                df.columns = df.columns.str.strip()
                st.sidebar.success("✅ Archivo cargado correctamente")
            except Exception as e:
                st.sidebar.error(f"Error al cargar archivo: {str(e)}")
    
    # Si no hay datos, cargar por defecto
    if df.empty:
        df = load_data()
    
    if df.empty:
        st.error("❌ No se pudieron cargar los datos")
        st.info("Por favor, verifica que el archivo 'Wyscout_League_Export.csv' existe o sube tu propio archivo")
        return
    
    # Navegación
    st.sidebar.markdown("---")
    st.sidebar.subheader("📍 Navegación")
    
    pages = {
        "📊 Explorador de Datos": page_data_explorer,
        "🕸️ Pizza Chart": page_pizza_chart,
        "📈 Scatter Plots": page_scatter_plots,
        "🔬 Análisis Avanzado": page_advanced_analysis
    }
    
    page = st.sidebar.radio(
        "Selecciona página:",
        list(pages.keys()),
        label_visibility="collapsed"
    )
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style='text-align: center'>
            <small>Dashboard v2.0 | Mejorado con IA</small>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Mostrar página seleccionada
    pages[page](df)

if __name__ == "__main__":
    main() 