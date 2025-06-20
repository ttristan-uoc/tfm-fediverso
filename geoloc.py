import plotly.express as px
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from adjustText import adjust_text


def instances_map(df):
    """
    Obtención del mapa de las instancias del Fediverso a partir de los datos de fediverse.observer
    Las 10 con más usuarios aparecen marcadas
    """

    # Lista de instancias importantes
    top10 = df.nlargest(10, 'total_users')
    relevant_instances = top10['domain'].tolist()

    # Crear GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['long'], df['lat']),
        crs="EPSG:4326"  # WGS84
    ).to_crs(epsg=3857)  # Proyección compatible con tiles de mapas

    # Colores por software
    software_colors = {
        'mastodon': 'purple',
        'peertube': 'red',
        'misskey': 'blue',
        'pleroma': 'orange',
        'akkoma': 'pink',
        'sharkey': 'green',
        'friendica': 'brown'
    }

    gdf['color'] = gdf['softwarename'].str.lower().map(software_colors).fillna('lightgray')

    # Crear la figura
    fig, ax = plt.subplots(figsize=(16, 10))

    # Filtrar solo los softwares con colores definidos
    valid_softwares = list(software_colors.keys())
    filtered_gdf = gdf[gdf['softwarename'].str.lower().isin(valid_softwares)]

    # Dibujar puntos por software con color definido
    for software, data in filtered_gdf.groupby('softwarename'):
        color = software_colors[software.lower()]
        data.plot(ax=ax, markersize=20, color=color, label=software, alpha=0.3)

    # Etiquetas para instancias relevantes (usando adjustText)
    texts = []
    relevant = gdf[gdf['domain'].isin(relevant_instances)]
    for _, row in relevant.iterrows():
        text = ax.text(
            row.geometry.x, row.geometry.y,
            f"{row['domain']}\n{row['total_users']} users",
            fontsize=8,
            color='black',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9)
        )
        texts.append(text)

    # Ajustar automáticamente las posiciones de las etiquetas
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    # Añadir fondo de mapa
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.VoyagerNoLabels)

    # Formato final
    ax.set_axis_off()
    plt.legend(title="Software", loc='best')

    # Guardar como PNG
    plt.tight_layout()
    plt.savefig("fediverso_mapa.png", dpi=500)
    plt.close()


def instances_heatmap(df):
    """
    Obtención del mapa de calor de las instancias del Fediverso a partir de los datos de fediverse.observer
    """
    # Agrupar por país para contar instancias
    country_counts = df.groupby("countryname").size().reset_index(name="num_instancias")
    country_counts = country_counts[country_counts["num_instancias"] > 0]

    # Crear el mapa de calor
    fig = px.choropleth(
        country_counts,
        locations="countryname",  # nombre del país
        locationmode="country names",  # modo de localización
        color="num_instancias",  # valor a colorear
        color_continuous_scale="OrRd",  # paleta de colores
        title="Número de instancias del Fediverso por país"
    )

    fig.update_traces(
        text=country_counts  # Solo mostrar el número
    )

    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True),
        coloraxis_colorbar=dict(title="Instancias")
    )

    # Guardar como HTML (interactivo)
    fig.write_html("fediverso_mapa_calor_pais.html")