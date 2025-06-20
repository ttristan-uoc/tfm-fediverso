import networkx as nx
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import powerlaw
import community.community_louvain as community_louvain
from collections import defaultdict
import os
import gc
from matplotlib.ticker import ScalarFormatter
import matplotlib.colors as mcolors
import warnings
import time
warnings.filterwarnings('ignore')

# Configuración para gráficos
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


class FediverseAnalyzer:
    def __init__(self, nodes_file, edges_file):
        """
        Inicializa el analizador de grafo del Fediverso

        Args:
            nodes_file (str): Ruta al archivo con los nodos (instancias)
            edges_file (str): Ruta al archivo JSON con las conexiones entre instancias
        """
        self.nodes_file = nodes_file
        self.edges_file = edges_file
        self.G = nx.Graph()  # Grafo dirigido
        self.valid_instances = set()  # Conjunto de instancias válidas

    def load_instances(self):
        """Carga las instancias desde el dataset de nodos"""
        print("Cargando instancias...")
        # Asumimos que el archivo de nodos es un CSV o similar
        # Ajusta según el formato real de tus datos
        df_instances = pd.read_csv(self.nodes_file)

        # Extraemos los dominios y los agregamos al conjunto de instancias válidas
        self.valid_instances = set(df_instances['domain'].values)

        # Añadimos los nodos al grafo con sus atributos
        for _, row in tqdm(df_instances.iterrows(), total=len(df_instances)):
            self.G.add_node(
                row['domain'],
                software=row.get('software', ''),
                users=row.get('users', 0),
                signup=row.get('signup', False)
            )

        print(f"Se cargaron {len(self.valid_instances)} instancias válidas")
        return self

    def build_graph_from_json(self):
        """Método alternativo: carga completa del JSON en memoria si es posible"""
        print("Cargando conexiones desde JSON completo...")
        try:
            with open(self.edges_file, 'r') as f:
                connections_dict = json.load(f)

            edge_count = 0
            # Solo añadimos aristas entre instancias válidas
            for source, targets in tqdm(connections_dict.items(), desc="Procesando conexiones"):
                if source in self.valid_instances:
                    for target in targets:
                        if target in self.valid_instances:
                            self.G.add_edge(source, target)
                            edge_count += 1

            print(f"Grafo construido con {len(self.G.nodes())} nodos y {edge_count} aristas")
        except (MemoryError, json.JSONDecodeError) as e:
            print(f"Error cargando JSON completo: {e}")
            print("Intenta usar el método load_connections() en su lugar")

        return self

    def clean_nodes(self):
        """Elimina nodos sin información de conexiones del grafo"""
        print("\n===== LIMPIEZA DE NODOS =====")

        # Identificar nodos sin información
        all_nodes = set(self.G.nodes())
        nodes_with_connections = set()
        with open(self.edges_file, 'r') as f:
            edges_data = json.load(f)

        nodes_with_connections.update(edges_data.keys())
        nodes_to_delete = all_nodes - nodes_with_connections

        print(f"Encontrados {len(nodes_to_delete)} nodos sin información")
        print(f"Grafo antes de la limpieza: {len(self.G.nodes())} nodos, {len(self.G.edges())} aristas")

        # Eliminar los nodos encontrados
        self.G.remove_nodes_from(nodes_to_delete)

        print(f"Grafo después de la limpieza: {len(self.G.nodes())} nodos, {len(self.G.edges())} aristas")

        return nodes_to_delete

    def export_cleaned_nodes_to_csv(self, output_file='cleaned_nodes.csv'):
        """
        Exporta los nodos que permanecen en el grafo después de la limpieza a un archivo CSV.
        Este archivo contiene solo los nodos que tienen información de conexiones.

        Args:
            output_file (str): Ruta del archivo CSV de salida

        Returns:
            str: Ruta del archivo generado
        """
        print(f"\n===== EXPORTANDO NODOS LIMPIOS A CSV =====")

        # Obtener los nodos actuales del grafo (después de la limpieza)
        remaining_nodes = set(self.G.nodes())
        print(f"Exportando {len(remaining_nodes)} nodos al CSV...")

        # Cargar el CSV original para mantener todos los atributos
        nodes_df = pd.read_csv(self.nodes_file)

        # Filtrar solo los nodos que permanecen en el grafo
        # Asumiendo que hay una columna 'id' o similar que identifica al nodo
        # Adaptar 'id' al nombre de columna correcto en tu CSV
        id_column = 'domain'  # Cambiar al nombre correcto de la columna que identifica al nodo

        # Filtrar el DataFrame para mantener solo los nodos con conexiones
        cleaned_df = nodes_df[nodes_df[id_column].isin(remaining_nodes)]

        # Guardar el DataFrame filtrado como CSV
        cleaned_df.to_csv(output_file, index=False)

        print(f"CSV generado exitosamente: {output_file}")
        print(f"Nodos en el CSV original: {len(nodes_df)}")
        print(f"Nodos en el CSV limpio: {len(cleaned_df)}")

        return output_file

    def print_basic_stats(self):
        """Muestra estadísticas básicas del grafo"""
        print("\n===== ESTADÍSTICAS BÁSICAS DEL GRAFO =====")
        print(f"Número de nodos (instancias): {len(self.G.nodes())}")
        print(f"Número de aristas (conexiones): {len(self.G.edges())}")

        # Grado promedio (en un grafo no dirigido, solo hay un tipo de grado)
        degrees = [d for _, d in self.G.degree()]

        print(f"Grado promedio: {np.mean(degrees):.2f}")

        # Densidad
        density = nx.density(self.G)
        print(f"Densidad del grafo: {density:.6f}")

        # Componentes (en un grafo no dirigido, solo hay componentes conexos)
        components = list(nx.connected_components(self.G))

        print(f"Número de componentes conexos: {len(components)}")
        print(f"Tamaño del componente conexo más grande: {len(max(components, key=len))}")

        return self

    def analyze_microscopic(self):
        """Análisis a nivel microscópico"""
        print("\n===== ANÁLISIS MICROSCÓPICO =====")

        # Calculamos métricas de centralidad para el componente más grande
        # para reducir el costo computacional
        print("Identificando el componente gigante...")
        giant_component = max(nx.connected_components(self.G), key=len)
        G_gc = self.G.subgraph(giant_component).copy()
        print(f"Componente gigante con {len(G_gc.nodes())} nodos y {len(G_gc.edges())} aristas")

        # Distribución de grados
        plt.figure(figsize=(12, 8))
        degrees = [d for _, d in self.G.degree()]

        plt.hist(degrees, bins=50, alpha=0.7, log=True)
        plt.title('Distribución de Grados (escala logarítmica)')
        plt.xlabel('Grado')
        plt.ylabel('Frecuencia (log)')
        plt.grid(True, alpha=0.3)
        plt.savefig('degree_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Análisis de ley de potencia para la distribución de grados
        print("Analizando distribución de grado (ley de potencia)...")
        # Filtramos valores <= 0 que powerlaw no puede manejar
        degrees_filtered = [d for d in degrees if d > 0]

        if not degrees_filtered:
            print("No hay grados positivos para analizar la ley de potencia")
            return self

        fit = powerlaw.Fit(degrees_filtered, verbose=False)
        alpha = fit.power_law.alpha
        print(f"Exponente de ley de potencia (alpha): {alpha:.3f}")

        # Usamos directamente los métodos de powerlaw para graficar CCDF
        plt.figure(figsize=(12, 8))
        fig, ax = plt.subplots(figsize=(12, 8))

        # Datos empíricos
        fit.plot_ccdf(ax=ax, color='b', linestyle='-', linewidth=2.5, label='Datos empíricos (ccdf)')

        # Distribuciones
        fit.power_law.plot_ccdf(ax=ax, color='r', linestyle='--', linewidth=2, label=f'Ley de potencia (α={alpha:.3f})')
        fit.exponential.plot_ccdf(ax=ax, linestyle=':', linewidth=2, label=f'Exponential')
        fit.lognormal.plot_ccdf(ax=ax, linestyle=':', linewidth=2, label=f'Lognormal')
        fit.truncated_power_law.plot_ccdf(ax=ax, linestyle=':', linewidth=2, label=f'Truncated Power Law')
        fit.stretched_exponential.plot_ccdf(ax=ax, linestyle=':', linewidth=2, label=f'Stretched Exponential')

        ax.set_xscale('log')
        ax.set_yscale('log')

        plt.title('Distribución de Grados vs. Ley de Potencia y otras distribuciones (CCDF)')
        plt.xlabel('Grado (k)')
        plt.ylabel('P(X ≥ k)')
        plt.grid(True, alpha=0.3, which='both')
        plt.legend()
        plt.tight_layout()
        plt.savefig('power_law_fit_ccdf.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Guardamos información adicional sobre el ajuste
        print(f"Valor mínimo para el ajuste (xmin): {fit.power_law.xmin}")
        print(f"Test Kolmogorov-Smirnov D: {fit.power_law.D}")

        # Comparamos con otras distribuciones
        distributions = ['exponential', 'lognormal', 'truncated_power_law', 'stretched_exponential']
        print("\nComparación con otras distribuciones:")
        for dist in distributions:
            try:
                R, p = fit.distribution_compare('power_law', dist)
                better = "ley de potencia" if R > 0 else dist
                print(f"  - Power law vs {dist}: R={R:.4f}, p={p:.4f} (Mejor: {better})")
            except Exception as e:
                print(f"  - Error comparando con {dist}: {e}")

        # Calculamos centralidades en el componente gigante
        print("Calculando centralidades (esto puede tomar tiempo)...")

        # Para grafos grandes, limitamos el análisis a una muestra
        if len(G_gc) > 5000:
            print(f"El componente gigante es muy grande ({len(G_gc)} nodos). Tomando una muestra de 5000 nodos.")
            sample_nodes = np.random.choice(list(G_gc.nodes()), 5000, replace=False)
            G_sample = G_gc.subgraph(sample_nodes).copy()
        else:
            G_sample = G_gc

        # Centralidad de grado (único en grafos no dirigidos)
        degree_centrality = nx.degree_centrality(G_sample)

        # Centralidad de cercanía (closeness)
        print("Calculando centralidad de cercanía...")
        closeness_centrality = nx.closeness_centrality(G_sample)

        # Centralidad de intermediación (betweenness)
        print("Calculando centralidad de intermediación (esto puede tomar tiempo)...")
        # Usamos k=1000 para aproximar en grafos grandes
        if len(G_sample) > 1000:
            betweenness_centrality = nx.betweenness_centrality(G_sample, k=1000, normalized=True)
        else:
            betweenness_centrality = nx.betweenness_centrality(G_sample, normalized=True)

        # Coeficiente de clustering
        print("Calculando coeficiente de clustering...")
        clustering = nx.clustering(G_sample)  # Ya no necesitamos to_undirected()
        c = clustering.values()
        average_clustering_coef = sum(c) / len(c)
        print(f"Coeficiente de clustering global: {average_clustering_coef:.4f}")

        # Visualizaciones de centralidades
        self._plot_centrality_distribution(degree_centrality.values(),
                                           'Centralidad de Grado',
                                           'degree_centrality.png')

        self._plot_centrality_distribution(closeness_centrality.values(),
                                           'Centralidad de Cercanía',
                                           'closeness_centrality.png')

        self._plot_centrality_distribution(betweenness_centrality.values(),
                                           'Centralidad de Intermediación',
                                           'betweenness_centrality.png')

        self._plot_centrality_distribution(clustering.values(),
                                           'Coeficiente de Clustering',
                                           'clustering_coefficient.png')

        # Identificación de hubs
        print("\nIdentificando hubs principales:")
        top_hubs_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:20]
        print("Top 20 hubs por centralidad de grado:")
        for node, centrality in top_hubs_degree:
            print(f"  - {node}: {centrality:.4f}")

        top_hubs_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:20]
        print("\nTop 20 hubs por centralidad de intermediación:")
        for node, centrality in top_hubs_betweenness:
            print(f"  - {node}: {centrality:.4f}")

        top_hubs_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:20]
        print("\nTop 20 hubs por centralidad de cercanía:")
        for node, centrality in top_hubs_closeness:
            print(f"  - {node}: {centrality:.4f}")

        top_hubs_clustering = sorted(clustering.items(), key=lambda x: x[1], reverse=True)[:20]
        print("\nTop 20 hubs por coeficiente de clustering:")
        for node, centrality, in top_hubs_clustering:
            print(f"  - {node}: {centrality:.4f}")

        return self

    def analyze_mesoscopic(self):
        """Análisis a nivel mesoscópico"""
        print("\n===== ANÁLISIS MESOSCÓPICO =====")

        # Trabajamos con el componente gigante para la detección de comunidades
        giant_component = max(nx.connected_components(self.G), key=len)
        G_gc = self.G.subgraph(giant_component).copy()
        print(f"Realizando análisis mesoscópico en componente gigante con {len(G_gc.nodes())} nodos")

        # Ya no necesitamos convertir a grafo no dirigido, pues ya lo es
        G_for_community = G_gc

        # Detección de comunidades con el algoritmo de Louvain
        print("Detectando comunidades con algoritmo Louvain (esto puede tomar tiempo)...")

        if community_louvain is None:
            print(
                "No se pudo ejecutar la detección de comunidades debido a que el módulo 'community' no está disponible")
            print("Instale python-louvain con: pip install python-louvain")
            return self

        # Intentamos detectar comunidades con el algoritmo de Louvain
        try:
            partition = community_louvain.best_partition(G_for_community)

            # Número de comunidades y sus tamaños
            communities = defaultdict(list)
            for node, community_id in partition.items():
                communities[community_id].append(node)

            print(f"Número de comunidades detectadas: {len(communities)}")

            # Tamaños de las comunidades
            community_sizes = [len(nodes) for _, nodes in communities.items()]

            plt.figure(figsize=(12, 8))
            plt.hist(community_sizes, bins=30, alpha=0.7)
            plt.title('Distribución de Tamaños de Comunidades')
            plt.xlabel('Tamaño de la Comunidad')
            plt.ylabel('Frecuencia')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.savefig('community_sizes.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Top comunidades por tamaño
            print("\nPrincipales comunidades por tamaño:")
            top_communities = sorted([(comm_id, len(nodes)) for comm_id, nodes in communities.items()],
                                     key=lambda x: x[1], reverse=True)

            for i, (comm_id, size) in enumerate(top_communities[:10]):
                print(f"  {i + 1}. Comunidad {comm_id}: {size} nodos")
                # Mostrar algunas instancias de ejemplo de esta comunidad
                sample_nodes = communities[comm_id][:5]  # Primeras 5 instancias
                print(f"     Ejemplos: {', '.join(sample_nodes)}")

            # Modularidad
            print("\nCalculando modularidad...")
            modularity = community_louvain.modularity(partition, G_for_community)
            print(f"Modularidad de la partición: {modularity:.4f}")

            # NUEVA VISUALIZACIÓN 1: Diagrama de red con comunidades coloreadas
            print("\nCreando visualización de grafo con comunidades...")

            # Si el grafo es muy grande, tomamos una muestra para la visualización
            if len(G_gc) > 500:
                # Seleccionamos las comunidades más grandes
                top_comm_ids = [comm_id for comm_id, _ in top_communities[:5]]
                nodes_to_include = []
                for comm_id in top_comm_ids:
                    # Tomamos hasta 100 nodos de cada comunidad grande
                    sample_size = min(100, len(communities[comm_id]))
                    nodes_to_include.extend(communities[comm_id][:sample_size])

                G_viz = G_for_community.subgraph(nodes_to_include).copy()
                print(f"Visualizando subgrafo con {len(G_viz.nodes())} nodos de las 5 comunidades principales")
            else:
                G_viz = G_for_community

            # Preparamos el layout y los colores
            plt.figure(figsize=(16, 16))
            pos = nx.spring_layout(G_viz, seed=42)

            # Colores para las comunidades
            cmap = plt.cm.rainbow
            comm_ids = set(partition.values())
            colors = [cmap(i / len(comm_ids)) for i in range(len(comm_ids))]
            color_map = {comm_id: colors[i % len(colors)] for i, comm_id in enumerate(comm_ids)}

            # Dibujamos los nodos coloreados por comunidad
            for comm_id in sorted(communities.keys()):
                nodes_in_comm = [node for node in G_viz.nodes() if partition.get(node) == comm_id]
                if nodes_in_comm:
                    nx.draw_networkx_nodes(G_viz, pos,
                                           nodelist=nodes_in_comm,
                                           node_color=[color_map[comm_id]],
                                           node_size=50,
                                           alpha=0.8,
                                           label=f"Comunidad {comm_id}")

            # Dibujamos las aristas
            nx.draw_networkx_edges(G_viz, pos, alpha=0.2)

            # Si el grafo no es muy grande, añadimos etiquetas
            if len(G_viz) <= 100:
                nx.draw_networkx_labels(G_viz, pos, font_size=8)

            plt.title('Visualización de Comunidades')
            plt.axis('off')
            plt.legend(scatterpoints=1, loc='lower right', ncol=2)
            plt.tight_layout()
            plt.savefig('community_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()

            # NUEVA VISUALIZACIÓN 2: Matriz de adyacencia ordenada por comunidades
            print("Creando matriz de adyacencia ordenada por comunidades...")

            # Limitamos a un subconjunto manejable si es necesario
            if len(G_viz) > 1000:
                print("La matriz de adyacencia sería demasiado grande. Limitando visualización.")
            else:
                # Ordenamos los nodos por comunidad
                ordered_nodes = []
                for comm_id in sorted(communities.keys()):
                    nodes_in_comm = [node for node in G_viz.nodes() if partition.get(node) == comm_id]
                    ordered_nodes.extend(sorted(nodes_in_comm))

                # Creamos la matriz de adyacencia ordenada
                adj_matrix = nx.to_numpy_array(G_viz, nodelist=ordered_nodes)

                # Visualizamos la matriz
                plt.figure(figsize=(14, 12))
                plt.imshow(adj_matrix, cmap='Blues', origin='upper')

                # Añadimos líneas para separar las comunidades
                prev_comm = None
                y_pos = 0
                for i, node in enumerate(ordered_nodes):
                    curr_comm = partition.get(node)
                    if curr_comm != prev_comm and prev_comm is not None:
                        plt.axhline(y=y_pos - 0.5, color='red', linestyle='-', alpha=0.7, linewidth=1)
                        plt.axvline(x=y_pos - 0.5, color='red', linestyle='-', alpha=0.7, linewidth=1)
                    prev_comm = curr_comm
                    y_pos += 1

                plt.title('Matriz de Adyacencia Ordenada por Comunidades')
                plt.colorbar(label='Conexión')
                plt.tight_layout()
                plt.savefig('community_adjacency_matrix.png', dpi=300, bbox_inches='tight')
                plt.close()

            # NUEVA VISUALIZACIÓN 3: Gráfico de barras de las principales comunidades
            plt.figure(figsize=(14, 8))
            top_n = min(20, len(communities))  # Mostramos hasta 20 comunidades
            comm_ids = [comm_id for comm_id, _ in top_communities[:top_n]]
            comm_sizes = [size for _, size in top_communities[:top_n]]

            bars = plt.bar(range(top_n), comm_sizes, color=[color_map.get(comm_id, 'gray') for comm_id in comm_ids])

            plt.title('Tamaño de las Principales Comunidades')
            plt.xlabel('ID de Comunidad')
            plt.ylabel('Número de Nodos')
            plt.xticks(range(top_n), [f"C{comm_id}" for comm_id in comm_ids], rotation=45)
            plt.grid(axis='y', alpha=0.3)

            # Añadimos los valores encima de cada barra
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                         f'{int(height)}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig('top_communities_size.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error en la detección de comunidades: {e}")
            print("Intente con un algoritmo alternativo como el método de Clauset-Newman-Moore:")

            # Alternativa: usar algoritmo de NetworkX para la detección de comunidades
            try:
                print("Usando algoritmo alternativo de NetworkX para detección de comunidades...")
                communities_generator = nx.community.greedy_modularity_communities(G_for_community)
                communities_list = list(communities_generator)

                print(f"Número de comunidades detectadas: {len(communities_list)}")

                # Tamaños de las comunidades
                community_sizes = [len(comm) for comm in communities_list]

                plt.figure(figsize=(12, 8))
                plt.hist(community_sizes, bins=30, alpha=0.7)
                plt.title('Distribución de Tamaños de Comunidades (algoritmo alternativo)')
                plt.xlabel('Tamaño de la Comunidad')
                plt.ylabel('Frecuencia')
                plt.xscale('log')
                plt.yscale('log')
                plt.grid(True, alpha=0.3)
                plt.savefig('community_sizes_alt.png', dpi=300, bbox_inches='tight')
                plt.close()

                # Top comunidades por tamaño
                print("\nPrincipales comunidades por tamaño:")
                communities_with_id = [(i, comm) for i, comm in enumerate(communities_list)]
                top_communities = sorted(communities_with_id, key=lambda x: len(x[1]), reverse=True)

                for i, (comm_id, nodes) in enumerate(top_communities[:10]):
                    size = len(nodes)
                    print(f"  {i + 1}. Comunidad {comm_id}: {size} nodos")
                    # Mostrar algunas instancias de ejemplo
                    sample_nodes = list(nodes)[:5]
                    print(f"     Ejemplos: {', '.join(sample_nodes)}")

                # NUEVA VISUALIZACIÓN: Grafo con comunidades para método alternativo
                print("\nCreando visualización de grafo con comunidades (método alternativo)...")

                # Si el grafo es muy grande, tomamos una muestra
                if len(G_gc) > 500:
                    # Tomamos los nodos de las comunidades más grandes
                    nodes_to_include = []
                    for _, comm in top_communities[:5]:
                        sample_size = min(100, len(comm))
                        nodes_to_include.extend(list(comm)[:sample_size])

                    G_viz = G_for_community.subgraph(nodes_to_include).copy()
                    print(f"Visualizando subgrafo con {len(G_viz.nodes())} nodos de las 5 comunidades principales")
                else:
                    G_viz = G_for_community

                # Creamos un diccionario de partición similar al de Louvain
                partition = {}
                for comm_id, nodes in communities_with_id:
                    for node in nodes:
                        partition[node] = comm_id

                # Preparamos el layout y los colores
                plt.figure(figsize=(16, 16))
                pos = nx.spring_layout(G_viz, seed=42)

                # Colores para las comunidades
                cmap = plt.cm.rainbow
                comm_ids = set(partition.values())
                colors = [cmap(i / len(comm_ids)) for i in range(len(comm_ids))]
                color_map = {comm_id: colors[i % len(colors)] for i, comm_id in enumerate(comm_ids)}

                # Dibujamos los nodos coloreados por comunidad
                for comm_id in range(len(communities_list)):
                    nodes_in_comm = [node for node in G_viz.nodes() if partition.get(node) == comm_id]
                    if nodes_in_comm:
                        nx.draw_networkx_nodes(G_viz, pos,
                                               nodelist=nodes_in_comm,
                                               node_color=[color_map[comm_id]],
                                               node_size=50,
                                               alpha=0.8,
                                               label=f"Comunidad {comm_id}")

                # Dibujamos las aristas
                nx.draw_networkx_edges(G_viz, pos, alpha=0.2)

                # Si el grafo no es muy grande, añadimos etiquetas
                if len(G_viz) <= 100:
                    nx.draw_networkx_labels(G_viz, pos, font_size=8)

                plt.title('Visualización de Comunidades (método alternativo)')
                plt.axis('off')
                plt.legend(scatterpoints=1, loc='lower right', ncol=2)
                plt.tight_layout()
                plt.savefig('community_visualization_alt.png', dpi=300, bbox_inches='tight')
                plt.close()

                # La modularidad es calculada internamente por el algoritmo de detección
                print("\nLa modularidad se maximizó internamente durante la detección de comunidades.")

                # Intentamos calcular la modularidad
                try:
                    modularity = nx.community.modularity(G_for_community, communities_list)
                    print(f"Modularidad de la partición: {modularity:.4f}")
                except:
                    print("No se pudo calcular la modularidad explícitamente.")

            except Exception as e2:
                print(f"También falló el algoritmo alternativo: {e2}")
                print("Saltando la detección de comunidades para continuar con el análisis.")

        return self

    def analyze_macroscopic(self):
        """Análisis a nivel macroscópico"""
        print("\n===== ANÁLISIS MACROSCÓPICO =====")

        # Componentes
        components = list(nx.connected_components(self.G))

        # Tamaños de componentes
        sizes = [len(c) for c in components]

        print(f"Componentes conexos: {len(components)}")
        print(
            f"  - Tamaño del componente gigante: {max(sizes)} nodos ({max(sizes) / len(self.G.nodes()) * 100:.2f}% del total)")

        # Visualización de distribución de tamaños de componentes
        plt.figure(figsize=(12, 8))
        plt.hist(sizes, bins=30, alpha=0.7, log=True)
        plt.title('Distribución de Tamaños de Componentes Conexos')
        plt.xlabel('Tamaño del Componente')
        plt.ylabel('Frecuencia (log)')
        plt.grid(True, alpha=0.3)
        plt.savefig('component_sizes.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Análisis del componente gigante
        giant_component = max(components, key=len)
        G_gc = self.G.subgraph(giant_component).copy()

        # Diámetro y camino promedio (puede ser costoso computacionalmente)
        print("\nAnalizando propiedades de mundo pequeño en el componente gigante...")

        # Si el componente es muy grande, usamos una aproximación
        if len(G_gc) > 1000:
            print("Componente gigante muy grande, calculando diámetro y camino promedio aproximados")
            # Tomamos una muestra de nodos
            sample_size = min(1000, len(G_gc))
            sample_nodes = np.random.choice(list(G_gc.nodes()), sample_size, replace=False)

            # Calculamos caminos entre pares en la muestra
            path_lengths = []
            for i, source in enumerate(tqdm(sample_nodes, desc="Calculando caminos")):
                for target in sample_nodes[i + 1:]:
                    try:
                        path_length = nx.shortest_path_length(G_gc, source=source, target=target)
                        path_lengths.append(path_length)
                    except nx.NetworkXNoPath:
                        pass

            if path_lengths:
                approx_diameter = max(path_lengths)
                avg_path_length = np.mean(path_lengths)
                print(f"Diámetro aproximado: {approx_diameter}")
                print(f"Longitud de camino promedio aproximada: {avg_path_length:.4f}")
            else:
                print("No se encontraron caminos en la muestra")
        else:
            # Grafo pequeño, calculamos exactamente
            avg_path_length = nx.average_shortest_path_length(G_gc)
            diameter = nx.diameter(G_gc)
            print(f"Diámetro del componente gigante: {diameter}")
            print(f"Longitud de camino promedio: {avg_path_length:.4f}")

        # Asortatividad
        print("\nCalculando asortatividad por grado...")
        try:
            assortativity = nx.degree_assortativity_coefficient(G_gc)
            print(f"Coeficiente de asortatividad por grado: {assortativity:.4f}")
            if assortativity > 0:
                print("La red es asortativa: los nodos tienden a conectarse con otros de grado similar")
            else:
                print("La red es disasortativa: los nodos tienden a conectarse con otros de grado diferente")
        except Exception as e:
            print(f"Error calculando asortatividad: {e}")

        return self

    def _plot_centrality_distribution(self, centrality_values, title, filename):
        """Genera un gráfico de distribución de centralidad"""
        plt.figure(figsize=(12, 8))

        # Convertimos a lista y filtramos valores no-cero para escala logarítmica
        values = [v for v in centrality_values if v > 0]

        if not values:
            print(f"No hay valores positivos para graficar {title}")
            return

        plt.hist(values, bins=50, alpha=0.7, log=True)
        plt.title(f'Distribución de {title} (escala logarítmica)')
        plt.xlabel(title)
        plt.ylabel('Frecuencia (log)')
        plt.grid(True, alpha=0.3)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        # También graficamos en escala normal para valores pequeños
        plt.figure(figsize=(12, 8))
        plt.hist(list(centrality_values), bins=50, alpha=0.7)
        plt.title(f'Distribución de {title}')
        plt.xlabel(title)
        plt.ylabel('Frecuencia')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'normal_{filename}', dpi=300, bbox_inches='tight')
        plt.close()

    def save_graph(self, output_file='fediverse_graph.graphml'):
        """Guarda el grafo en formato GraphML"""
        print(f"\nGuardando grafo en {output_file}...")
        nx.write_graphml(self.G, output_file)
        print(f"Grafo guardado exitosamente en {output_file}")
        return self

    def visualize_network(self, output_file=None, min_connections=1,
                          width=1200, height=800, show_labels=True, label_cutoff=20,
                          node_size_factor=100, edge_opacity=0.2, node_opacity=0.4):
        """
        Visualiza el grafo del Fediverso usando colores para el software y tamaño para usuarios

        Args:
            output_file (str, optional): Ruta para guardar la imagen. Si es None, solo muestra.
            min_connections (int, optional): Número mínimo de conexiones para mostrar un nodo.
            top_instances (int, optional): Limitar visualización a las N instancias con más conexiones.
            width (int): Ancho de la figura en píxeles.
            height (int): Altura de la figura en píxeles.
            show_labels (bool): Si se muestran las etiquetas de los nodos.
            label_cutoff (int): Número mínimo de conexiones para mostrar etiqueta.
            node_size_factor (int): Factor para escalar el tamaño de los nodos.
            edge_opacity (float): Opacidad de las aristas (0-1).
            node_opacity (float): Opacidad de los nodos (0-1).

        Returns:
            self: Para encadenamiento de métodos
        """
        import matplotlib.colors as mcolors

        print("Preparando visualización del grafo...")

        # Filtrar nodos con menos conexiones que min_connections
        if min_connections > 1:
            view = self.G.copy()
            nodes_to_remove = [node for node, degree in view.degree()
                               if degree < min_connections]
            view.remove_nodes_from(nodes_to_remove)
        else:
            view = self.G

        if len(view.nodes()) == 0:
            print("No hay nodos que cumplan los criterios de filtrado")
            return self

        print(f"Visualizando grafo con {len(view.nodes())} nodos y {len(view.edges())} aristas")

        # Crear un mapa de colores para los diferentes tipos de software
        software_types = set()
        for _, attrs in view.nodes(data=True):
            if 'software' in attrs and attrs['software']:
                software_types.add(attrs['software'])

        # Usar un conjunto de colores predefinidos (evitando colores muy claros)
        colormap = plt.cm.tab20
        color_dict = {software: colormap(i % 20)
                      for i, software in enumerate(sorted(software_types))}
        # Color por defecto
        color_dict[''] = (0.7, 0.7, 0.7, 1.0)  # Gris

        # Escalar tamaños de nodos según número de usuarios
        sizes = []
        max_users = 1  # Evitar división por cero
        for node, attrs in view.nodes(data=True):
            users = attrs.get('users', 0)
            if isinstance(users, (int, float)) and users > max_users:
                max_users = users

        # Determinar colores y tamaños
        node_colors = []
        node_sizes = []
        for node, attrs in view.nodes(data=True):
            software = attrs.get('software', '')
            node_colors.append(color_dict.get(software, color_dict['']))

            # Calcular tamaño basado en usuarios (con un tamaño mínimo)
            users = attrs.get('users', 0)
            if not isinstance(users, (int, float)):
                users = 0
            # Escala logarítmica para que instancias pequeñas sigan siendo visibles
            size = node_size_factor * (np.log1p(users) / np.log1p(max_users)) + 5
            node_sizes.append(size)

        # Crear figura
        plt.figure(figsize=(width / 100, height / 100), dpi=100)

        # Posicionamiento del grafo - usar un layout que separe bien los nodos
        pos = nx.spring_layout(view, k=0.3, iterations=50, seed=42)

        # Dibujar aristas con transparencia
        nx.draw_networkx_edges(view, pos, alpha=edge_opacity, edge_color='gray')

        # Dibujar nodos
        nx.draw_networkx_nodes(view, pos, node_color=node_colors, node_size=node_sizes, alpha=node_opacity)

        # Añadir etiquetas sólo a nodos con suficientes conexiones
        if show_labels:
            labels = {}
            for node in view.nodes():
                if view.degree(node) >= label_cutoff:
                    labels[node] = node
            nx.draw_networkx_labels(view, pos, labels=labels, font_size=8)

        # Crear leyenda para tipos de software
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor=color_dict[software],
                                     markersize=8, label=software)
                          for software in sorted(color_dict.keys())]

        plt.legend(handles=legend_handles, loc='upper right',
                   title='Software', fontsize='small', bbox_to_anchor=(1.1, 1))

        plt.axis('off')
        plt.tight_layout()

        # Guardar o mostrar
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            print(f"Visualización guardada en: {output_file}")
        else:
            plt.show()

        return self

    def visualize_adjacency_matrix(self, output_file=None, top_instances=None,
                                   colormap='viridis', width=1000, height=900, show_tick_labels=True,
                                   tick_label_size=8, sort_by='degree'):
        """
        Visualiza la matriz de adyacencia del grafo ordenada por grados u otra métrica

        Args:
            output_file (str, optional): Ruta para guardar la imagen. Si es None, solo muestra.
            top_instances (int, optional): Limitar visualización a las N instancias con más conexiones.
            colormap (str): Nombre del mapa de colores de matplotlib a usar.
            width (int): Ancho de la figura en píxeles.
            height (int): Altura de la figura en píxeles.
            show_tick_labels (bool): Si se muestran las etiquetas de los ejes.
            tick_label_size (int): Tamaño de fuente para las etiquetas de los ejes.
            sort_by (str): Criterio para ordenar: 'degree' (default), 'in_degree', 'out_degree', 'users', 'software'.

        Returns:
            self: Para encadenamiento de métodos
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap

        print("Preparando matriz de adyacencia...")

        # Obtener la lista de nodos
        if top_instances is not None:
            degrees = dict(self.G.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_instances]
            nodes = [node for node, _ in top_nodes]
        else:
            nodes = list(self.G.nodes())

        if len(nodes) == 0:
            print("No hay nodos para visualizar")
            return self

        # Ordenar nodos según el criterio especificado
        if sort_by == 'degree':
            nodes.sort(key=lambda x: self.G.degree(x), reverse=True)
        elif sort_by == 'users':
            nodes.sort(key=lambda x: self.G.nodes[x].get('users', 0)
            if isinstance(self.G.nodes[x].get('users', 0), (int, float)) else 0,
                       reverse=True)
        elif sort_by == 'software':
            nodes.sort(key=lambda x: self.G.nodes[x].get('software', ''))

        # Crear matriz de adyacencia ordenada
        adj_matrix = np.zeros((len(nodes), len(nodes)))
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        for i, source in enumerate(nodes):
            for target in self.G.neighbors(source):
                if target in node_to_idx:  # Asegurarse de que el destino está en nuestra lista filtrada
                    j = node_to_idx[target]
                    adj_matrix[i, j] = 1

        # Crear figura
        plt.figure(figsize=(width / 100, height / 100), dpi=100)

        # Personalizar el colormap para que el cero sea blanco o muy claro
        if colormap == 'custom':
            # Crear un colormap personalizado que va de blanco a azul oscuro
            colors = [(1, 1, 1), (0, 0, 0.8)]  # Blanco a azul oscuro
            custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
            cmap = custom_cmap
        else:
            cmap = plt.get_cmap(colormap)

        # Visualizar la matriz
        plt.imshow(adj_matrix, cmap=cmap, interpolation='none')
        plt.colorbar(label='Conexión')

        # Configurar etiquetas si se solicita
        if show_tick_labels:
            # Acortar nombres de dominio para que sean más legibles en las etiquetas
            shortened_labels = []
            for node in nodes:
                if len(node) > 15:  # Acortar dominios muy largos
                    parts = node.split('.')
                    if len(parts) > 2:
                        shortened = parts[0][:10] + "..." + '.'.join(parts[-2:])
                    else:
                        shortened = node[:15] + "..."
                else:
                    shortened = node
                shortened_labels.append(shortened)

            plt.xticks(range(len(nodes)), shortened_labels, rotation=90, fontsize=tick_label_size)
            plt.yticks(range(len(nodes)), shortened_labels, fontsize=tick_label_size)
        else:
            plt.xticks([])
            plt.yticks([])

        plt.title(f'Matriz de Adyacencia')
        plt.tight_layout()

        # Guardar o mostrar
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            print(f"Matriz de adyacencia guardada en: {output_file}")
        else:
            plt.show()

        return self

    def plot_betweenness_degree_correlation(self, output_file=None,
                                            width=900, height=700, annotate_top=10,
                                            colormap='viridis',
                                            highlight_software=None, log_scale=True):
        """
        Visualiza la correlación entre la centralidad de intermediación (betweenness)
        y el grado (número de conexiones) de los nodos

        Args:
            output_file (str, optional): Ruta para guardar la imagen. Si es None, solo muestra.
            width (int): Ancho de la figura en píxeles.
            height (int): Altura de la figura en píxeles.
            annotate_top (int): Cantidad de nodos a etiquetar (los de mayor betweenness).
            colormap (str): Nombre del mapa de colores para codificar el software.
            highlight_software (str, optional): Nombre del software a destacar.
            log_scale (bool): Si se usa escala logarítmica para los ejes.

        Returns:
            self: Para encadenamiento de métodos
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm
        import time

        print("Calculando centralidad de intermediación...")
        start_time = time.time()

        view = self.G

        # Calcular centralidad de intermediación (puede ser computacionalmente intensivo)
        betweenness = nx.betweenness_centrality(view)

        # Preparar los datos para el gráfico
        node_degrees = dict(view.degree())

        end_time = time.time()
        print(f"Cálculo completado en {end_time - start_time:.2f} segundos")

        # Extraer software para colorear los puntos
        software_types = set()
        for node in betweenness.keys():
            if 'software' in view.nodes[node] and view.nodes[node]['software']:
                software_types.add(view.nodes[node]['software'])

        software_index = {software: i for i, software in enumerate(sorted(software_types))}
        software_index[''] = len(software_index)  # Para nodos sin software identificado

        # Preparar datos para el gráfico
        x_values = []  # Grados
        y_values = []  # Betweenness
        colors = []  # Colores según software
        labels = []  # Nombres de los nodos
        software_list = []  # Lista de software para cada nodo

        # Colormap para los diferentes tipos de software
        cmap = plt.get_cmap(colormap)
        color_norm = Normalize(vmin=0, vmax=max(1, len(software_types)))

        for node, centrality in betweenness.items():
            x_values.append(node_degrees[node])
            y_values.append(centrality)
            labels.append(node)

            # Obtener el software del nodo
            if 'software' in view.nodes[node] and view.nodes[node]['software']:
                node_software = view.nodes[node]['software']
            else:
                node_software = ''

            software_list.append(node_software)

            # Asignar color según software
            if highlight_software and node_software == highlight_software:
                # Si estamos destacando un software específico, usamos un color fijo para él
                colors.append('red')
            else:
                idx = software_index.get(node_software, len(software_index))
                colors.append(cmap(color_norm(idx)))

        # Crear figura
        plt.figure(figsize=(width / 100, height / 100), dpi=100)

        # Configurar escala logarítmica si se solicita
        if log_scale:
            plt.xscale('log')
            plt.yscale('log')

        # Crear scatter plot
        scatter = plt.scatter(x_values, y_values, c=colors, alpha=0.7, s=50)

        # Calcular y dibujar la línea de tendencia (regresión)
        if len(x_values) > 1:
            # Convertir a log si estamos usando escala log
            if log_scale:
                x_reg = np.log10(np.array(x_values))
                y_reg = np.log10(np.array(y_values))
            else:
                x_reg = np.array(x_values)
                y_reg = np.array(y_values)

            # Evitar -inf en log(0)
            x_reg = x_reg[np.isfinite(x_reg) & np.isfinite(y_reg)]
            y_reg = y_reg[np.isfinite(x_reg) & np.isfinite(y_reg)]

            if len(x_reg) > 1:
                # Calcular regresión lineal
                coeffs = np.polyfit(x_reg, y_reg, 1)
                poly = np.poly1d(coeffs)

                # Generar puntos para la línea de tendencia
                if log_scale:
                    x_trend = np.logspace(np.log10(min(x_values)), np.log10(max(x_values)), 100)
                    y_trend = 10 ** poly(np.log10(x_trend))
                else:
                    x_trend = np.linspace(min(x_values), max(x_values), 100)
                    y_trend = poly(x_trend)

                plt.plot(x_trend, y_trend, 'r--', alpha=0.7)

                # Mostrar la ecuación en el gráfico
                if log_scale:
                    equation = f"log(Betweenness) = {coeffs[0]:.2f} × log(Grado) + {coeffs[1]:.2f}"
                else:
                    equation = f"Betweenness = {coeffs[0]:.6f} × Grado + {coeffs[1]:.6f}"

                plt.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction',
                             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        # Anotar los nodos top según centralidad de intermediación
        if annotate_top > 0:
            top_centrality_indices = np.argsort(y_values)[-annotate_top:]
            for idx in top_centrality_indices:
                plt.annotate(labels[idx],
                             (x_values[idx], y_values[idx]),
                             xytext=(5, 5),
                             textcoords='offset points',
                             fontsize=8,
                             bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7))

        # Crear leyenda para tipos de software
        if highlight_software:
            # Si estamos destacando un software, simplificamos la leyenda
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                                          markersize=8, label=highlight_software),
                               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0.5),
                                          markersize=8, label='Otros')]
        else:
            # Crear un conjunto ordenado de software único que aparece en los datos
            unique_software = sorted(set(software_list))
            if '' in unique_software:
                unique_software.remove('')
                unique_software.append('')  # Mover "desconocido" al final

            legend_elements = []
            for software in unique_software:
                idx = software_index.get(software, len(software_index))
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=cmap(color_norm(idx)),
                               markersize=8,
                               label=software)
                )

        plt.legend(handles=legend_elements, loc='upper left',
                   title='Software', fontsize='small')

        plt.xlabel('Grado')
        plt.ylabel('Centralidad de Intermediación')
        plt.title('Correlación entre Centralidad de Intermediación y Grado')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Guardar o mostrar
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            print(f"Gráfico guardado en: {output_file}")
        else:
            plt.show()

        return self

    def visualize_nested_circle_plot(self, output_file=None, num_shells=5, min_degree=1,
                                     width=1200, height=1200, show_labels=True, label_size=8,
                                     label_min_degree=10, edge_opacity=0.15, node_size_factor=30,
                                     colormap='tab20'):
        """
        Visualiza la estructura anidada (nestedness) del Fediverso usando un gráfico circular
        con nodos organizados en coronas/shells según su grado de conectividad

        Args:
            output_file (str, optional): Ruta para guardar la imagen. Si es None, solo muestra.
            num_shells (int): Número de coronas o shells para la visualización.
            min_degree (int): Grado mínimo de nodos a incluir.
            width (int): Ancho de la figura en píxeles.
            height (int): Altura de la figura en píxeles.
            show_labels (bool): Si se muestran las etiquetas de los nodos.
            label_size (int): Tamaño de fuente para las etiquetas.
            label_min_degree (int): Grado mínimo para que un nodo muestre su etiqueta.
            edge_opacity (float): Opacidad de las aristas (0-1).
            node_size_factor (float): Factor para escalar el tamaño de los nodos.
            colormap (str): Nombre del mapa de colores para software.

        Returns:
            self: Para encadenamiento de métodos
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import math
        from matplotlib.colors import Normalize

        print("Preparando visualización circular anidada...")

        # Filtrar nodos con grado mínimo
        if min_degree > 1:
            view = self.G.copy()
            nodes_to_remove = [node for node, degree in view.degree()
                               if degree < min_degree]
            view.remove_nodes_from(nodes_to_remove)
        else:
            view = self.G

        if len(view.nodes()) == 0:
            print("No hay nodos que cumplan el criterio de grado mínimo")
            return self

        # Obtener los grados de los nodos
        node_degrees = dict(view.degree())

        # Determinar rangos de grados para cada corona/shell
        min_deg = min(node_degrees.values())
        max_deg = max(node_degrees.values())

        # Calcular intervalos de grado para cada corona
        # Usamos escala logarítmica para que la distribución sea más uniforme
        if min_deg == max_deg:
            # Si todos los nodos tienen el mismo grado, usamos una sola corona
            shell_ranges = [(min_deg, max_deg)]
        else:
            if min_deg > 0:
                log_min = math.log(min_deg)
            else:
                log_min = 0
            log_max = math.log(max_deg)
            log_step = (log_max - log_min) / num_shells

            shell_ranges = []
            for i in range(num_shells):
                if i == num_shells - 1:
                    # Para la última corona, incluimos el valor máximo
                    shell_ranges.append((math.ceil(math.exp(log_min + i * log_step)),
                                         math.ceil(math.exp(log_min + (i + 1) * log_step))))
                else:
                    shell_ranges.append((math.ceil(math.exp(log_min + i * log_step)),
                                         math.floor(math.exp(log_min + (i + 1) * log_step))))

        # Distribuir nodos en las coronas según su grado
        shells = [[] for _ in range(len(shell_ranges))]
        for node, degree in node_degrees.items():
            for i, (min_shell_deg, max_shell_deg) in enumerate(shell_ranges):
                if min_shell_deg <= degree <= max_shell_deg:
                    shells[i].append(node)
                    break

        # Eliminar coronas vacías
        shells = [shell for shell in shells if shell]

        # Ordenar nodos dentro de cada corona para mejor visualización
        # Los ordenamos por software para que nodos del mismo tipo estén juntos
        for i, shell in enumerate(shells):
            shells[i] = sorted(shell, key=lambda x: view.nodes[x].get('software', ''))

        # Configurar colores basados en software
        software_types = set()
        for _, attrs in view.nodes(data=True):
            if 'software' in attrs and attrs['software']:
                software_types.add(attrs['software'])

        # Crear diccionario de colores para software
        cmap = plt.cm.get_cmap(colormap)
        color_dict = {software: cmap(i % cmap.N)
                      for i, software in enumerate(sorted(software_types))}
        color_dict[''] = (0.7, 0.7, 0.7, 1.0)  # Color por defecto para desconocido

        # Preparar figura
        plt.figure(figsize=(width / 100, height / 100), dpi=100)

        # Crear layout circular anidado
        pos = nx.shell_layout(view, shells)

        # Preparar tamaños y colores de nodos
        node_colors = []
        node_sizes = []
        for node in view.nodes():
            # Color según software
            software = view.nodes[node].get('software', '')
            node_colors.append(color_dict.get(software, color_dict['']))

            # Tamaño según grado (con un valor mínimo)
            size = node_size_factor * math.log1p(node_degrees[node])
            node_sizes.append(max(size, 5))

        # Dibujar aristas con baja opacidad
        nx.draw_networkx_edges(view, pos, alpha=edge_opacity, edge_color='gray', width=0.5)

        # Dibujar nodos
        nx.draw_networkx_nodes(view, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)

        # Añadir etiquetas para nodos con grado suficiente
        if show_labels:
            labels = {}
            for node in view.nodes():
                if node_degrees[node] >= label_min_degree:
                    labels[node] = node
            nx.draw_networkx_labels(view, pos, labels=labels, font_size=label_size,
                                    font_family='sans-serif')

        # Dibujar círculos concéntricos para indicar las coronas
        # Calculamos radios para los círculos basados en el layout
        center_x = sum(pos[node][0] for node in view.nodes()) / len(view.nodes())
        center_y = sum(pos[node][1] for node in view.nodes()) / len(view.nodes())

        # Encontrar el radio máximo desde el centro hasta algún nodo
        max_radius = max(math.sqrt((pos[node][0] - center_x) ** 2 +
                                   (pos[node][1] - center_y) ** 2)
                         for node in view.nodes())

        # Dibujar círculos concéntricos para las coronas
        for i in range(1, len(shells) + 1):
            radius = max_radius * i / len(shells)
            circle = plt.Circle((center_x, center_y), radius, fill=False,
                                color='gray', linestyle='--', alpha=0.5)
            plt.gca().add_patch(circle)

        # Anotar los rangos de grado en cada corona
        for i, (min_shell_deg, max_shell_deg) in enumerate(shell_ranges[:len(shells)]):
            radius = max_radius * (i + 0.5) / len(shells)
            angle = 45  # Ángulo en grados para la etiqueta
            x = center_x + radius * math.cos(math.radians(angle))
            y = center_y + radius * math.sin(math.radians(angle))

            if min_shell_deg == max_shell_deg:
                label = f"Grado: {min_shell_deg}"
            else:
                label = f"Grados: {min_shell_deg}-{max_shell_deg}"

            plt.annotate(label, xy=(x, y), xytext=(x, y), fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                         ha='center', va='center')

        # Crear leyenda para tipos de software
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=color_dict[software],
                                      markersize=8, label=software)
                           for software in sorted(color_dict.keys())]

        # Solo mostramos leyenda si no hay demasiados tipos de software
        if len(legend_elements) <= 15:
            plt.legend(handles=legend_elements, loc='upper right',
                       title='Software', fontsize='small')

        plt.axis('off')
        plt.title('Estructura Anidada del Fediverso', fontsize=14)

        # Añadir una anotación explicativa
        plt.annotate(
            f"Distribución en {len(shells)} coronas\nOrdenadas por grado de conectividad",
            xy=(0.02, 0.02), xycoords='axes fraction',
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
        )

        plt.tight_layout()

        # Guardar o mostrar
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            print(f"Visualización circular anidada guardada en: {output_file}")
        else:
            plt.show()

        return self

    def calculate_and_save_centrality_metrics(self, output_file='centrality_metrics.csv'):
        """
        Calcula las métricas de centralidad para cada nodo del grafo y las guarda en un CSV.

        Métricas calculadas:
        - Grado
        - Centralidad de grado
        - Centralidad de cercanía
        - Centralidad de intermediación
        - Coeficiente de clústering

        Args:
            output_file (str): Ruta donde guardar el archivo CSV de salida.

        Returns:
            pd.DataFrame: DataFrame con las métricas calculadas.
        """
        print("Calculando métricas de centralidad...")

        # Inicializar un diccionario para almacenar las métricas
        metrics = {
            'node': [],
            'degree': [],
            'degree_centrality': [],
            'closeness_centrality': [],
            'betweenness_centrality': [],
            'clustering_coefficient': []
        }

        # Calcular las métricas para todo el grafo
        print("Calculando centralidad de grado...")
        degree_centrality = nx.degree_centrality(self.G)

        print("Calculando centralidad de cercanía...")
        # Para grafos no conectados, usamos la versión que maneja componentes desconectados
        closeness_centrality = nx.closeness_centrality(self.G)

        print("Calculando centralidad de intermediación...")
        # Esto puede ser computacionalmente intensivo para grafos grandes
        betweenness_centrality = nx.betweenness_centrality(self.G, normalized=True)

        print("Calculando coeficiente de clústering...")
        clustering_coefficient = nx.clustering(self.G)

        # Recopilar todas las métricas en el diccionario
        for node in tqdm(self.G.nodes(), desc="Recopilando métricas"):
            metrics['node'].append(node)
            metrics['degree'].append(self.G.degree(node))
            metrics['degree_centrality'].append(degree_centrality[node])
            metrics['closeness_centrality'].append(closeness_centrality[node])
            metrics['betweenness_centrality'].append(betweenness_centrality[node])
            metrics['clustering_coefficient'].append(clustering_coefficient[node])

        # Crear un DataFrame con las métricas
        df_metrics = pd.DataFrame(metrics)

        # Guardar el DataFrame como CSV
        df_metrics.to_csv(output_file, index=False)
        print(f"Métricas de centralidad guardadas en {output_file}")

        return df_metrics

    def plot_centrality_distributions(self, metrics_file='centrality_metrics.csv', output_dir='plots/'):
        """
        Genera gráficos de distribución para cada métrica de centralidad en formato
        de subplot doble: escala logarítmica y lineal en el eje vertical.

        Args:
            metrics_file (str): Ruta al archivo CSV con las métricas de centralidad
            output_dir (str): Directorio donde guardar las imágenes generadas

        Returns:
            dict: Diccionario con las figuras generadas para cada métrica
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os

        # Crear directorio de salida si no existe
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Cargar datos de métricas
        print(f"Cargando métricas desde {metrics_file}...")
        df_metrics = pd.read_csv(metrics_file)

        # Métricas a analizar (excluyendo 'node' que es el nombre del nodo)
        metrics = [col for col in df_metrics.columns if col != 'node']

        # Configuración estética para las gráficas
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
        })

        figures = {}
        metric_titles = ['Grado', 'Centralidad de grado', 'Centralidad de cercanía', 'Centralidad de intermediación',
                         'Coeficiente de clustering']
        # Crear un gráfico para cada métrica
        for i, metric in enumerate(metrics):
            print(f"Generando gráfico para: {metric_titles[i]}")

            # Filtrar valores > 0 para escala logarítmica (evitar -inf)
            # También eliminamos valores nulos
            df_filtered = df_metrics[df_metrics[metric] > 0].dropna(subset=[metric])

            if len(df_filtered) == 0:
                print(f"  Advertencia: No hay valores positivos para {metric_titles[i]}, omitiendo gráfico")
                continue

            # Crear figura con dos subplots verticales (lineal y logarítmico)
            fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True)

            # Título principal
            fig.suptitle(f'Distribución de {metric_titles[i]} en la red del Fediverso', fontsize=16)

            # Gráfico superior: Escala lineal
            sns.histplot(data=df_metrics, x=metric, kde=True, ax=axes[0],
                         color='steelblue', bins=30, stat='density', element='step')
            axes[0].set_title(f'Escala Lineal')
            axes[0].set_xlabel(f'{metric_titles[i]}')  # Ocultamos la etiqueta x del primer subplot
            axes[0].set_ylabel('Densidad')

            # Añadir estadísticas al gráfico
            stats_text = (
                f"Media: {df_metrics[metric].mean():.4f}\n"
                f"Mediana: {df_metrics[metric].median():.4f}\n"
                f"Máx: {df_metrics[metric].max():.4f}\n"
                f"Desv. Est.: {df_metrics[metric].std():.4f}"
            )
            axes[0].text(0.95, 0.95, stats_text, transform=axes[0].transAxes,
                         verticalalignment='top', horizontalalignment='left',
                         bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})

            # Gráfico inferior: Escala logarítmica
            sns.histplot(data=df_filtered, x=metric, kde=True, ax=axes[1],
                         color='darkred', bins=30, stat='density', element='step')
            axes[1].set_title(f'Escala Logarítmica')
            axes[1].set_ylabel('Densidad (log)')
            axes[1].set_yscale('log')
            axes[1].set_xlabel(f'{metric_titles[i]}')

            # Ajustar layout
            plt.tight_layout()

            # Guardar figura
            output_file = os.path.join(output_dir, f'{metric}_distribution.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  Guardado como: {output_file}")

            # Guardar referencia a la figura
            figures[metric] = fig

        # Crear un gráfico conjunto que muestre todas las métricas en un único pannel
        print("Generando gráfico conjunto de todas las métricas...")
        fig_all, axes_all = plt.subplots(len(metrics), 2, figsize=(14, 5 * len(metrics)))
        fig_all.suptitle('Distribuciones de métricas de centralidad en la red del Fediverso', fontsize=18)

        for i, metric in enumerate(metrics):
            df_filtered = df_metrics[df_metrics[metric] > 0].dropna(subset=[metric])

            if len(df_filtered) == 0:
                continue

            # Escala lineal (columna izquierda)
            sns.histplot(data=df_metrics, x=metric, kde=True, ax=axes_all[i, 0],
                         color='steelblue', bins=30, stat='density', element='step')
            axes_all[i, 0].set_title(f'{metric} - Lineal')
            axes_all[i, 0].set_ylabel('Densidad')

            # Escala logarítmica (columna derecha)
            sns.histplot(data=df_filtered, x=metric, kde=True, ax=axes_all[i, 1],
                         color='darkred', bins=30, stat='density', element='step')
            axes_all[i, 1].set_title(f'{metric} - Logarítmica')
            axes_all[i, 1].set_yscale('log')

            # Solo mostrar etiqueta x en la última fila
            if i < len(metrics) - 1:
                axes_all[i, 0].set_xlabel('')
                axes_all[i, 1].set_xlabel('')

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)

        # Guardar gráfico conjunto
        output_file_all = os.path.join(output_dir, 'all_metrics_distribution.png')
        plt.savefig(output_file_all, dpi=300, bbox_inches='tight')
        print(f"Gráfico conjunto guardado como: {output_file_all}")

        # Agregar el gráfico conjunto al diccionario de salida
        figures['all'] = fig_all

        print(f"Todos los gráficos generados en: {output_dir}")

        return figures

    def detect_and_compare_communities(self, output_file='community_comparison.csv', num_runs=1):
        """
        Detecta comunidades utilizando diferentes algoritmos de la biblioteca cdlib,
        evalúa sus modularidades y permite comparar los resultados.

        Args:
            output_file (str): Ruta donde guardar el archivo CSV con los resultados
            num_runs (int): Número de ejecuciones para algoritmos no deterministas

        Returns:
            dict: Diccionario con los resultados de detección de comunidades y métricas
        """
        try:
            import cdlib
            from cdlib import algorithms, evaluation
        except ImportError:
            print("Por favor, instala la biblioteca cdlib: pip install cdlib")
            return None

        print(
            f"Analizando estructura comunitaria del grafo ({len(self.G.nodes())} nodos, {len(self.G.edges())} aristas)...")

        # Resultados para guardar
        results = {
            'algorithm': [],
            'num_communities': [],
            'modularity': [],
            'avg_community_size': [],
            'min_community_size': [],
            'max_community_size': [],
            'execution_time': [],
            'communities_distribution': []
        }

        # Lista de algoritmos a probar
        # Elegimos algoritmos que escalen bien para grafos grandes
        algorithms_to_try = [
            ('Louvain', lambda g: algorithms.louvain(g, resolution=1.0, randomize=False)),
            ('Label Propagation', lambda g: algorithms.label_propagation(g)),
            ('Leiden', lambda g: algorithms.leiden(g)),
            ('Infomap', lambda g: algorithms.infomap(g)),
            # ('Greedy Modularity', lambda g: algorithms.greedy_modularity(g))
        ]

        # Ejecutar cada algoritmo
        for algo_name, algo_func in algorithms_to_try:
            try:
                print(f"\nEjecutando algoritmo: {algo_name}")

                # Medir tiempo de ejecución
                best_modularity = -1
                best_communities = None
                best_execution_time = float('inf')

                # Ejecutar múltiples veces para algoritmos no deterministas
                for run in range(num_runs):
                    if num_runs > 1:
                        print(f"  Ejecución {run + 1}/{num_runs}")

                    start_time = time.time()
                    communities = algo_func(self.G)
                    execution_time = time.time() - start_time

                    # Calcular modularidad
                    modularity = evaluation.newman_girvan_modularity(self.G, communities).score

                    if modularity > best_modularity:
                        best_modularity = modularity
                        best_communities = communities
                        best_execution_time = execution_time

                # Analizar las comunidades detectadas
                community_sizes = [len(comm) for comm in best_communities.communities]

                # Guardar resultados
                results['algorithm'].append(algo_name)
                results['num_communities'].append(len(best_communities.communities))
                results['modularity'].append(best_modularity)
                results['avg_community_size'].append(sum(community_sizes) / len(community_sizes))
                results['min_community_size'].append(min(community_sizes))
                results['max_community_size'].append(max(community_sizes))
                results['execution_time'].append(best_execution_time)

                # Distribución de tamaños de comunidades (para análisis posterior)
                # Guardamos los percentiles 25, 50, 75
                percentiles = np.percentile(community_sizes, [25, 50, 75])
                results['communities_distribution'].append(
                    f"25%: {percentiles[0]:.1f}, 50%: {percentiles[1]:.1f}, 75%: {percentiles[2]:.1f}"
                )

                print(f"  Completado - {len(best_communities.communities)} comunidades detectadas")
                print(f"  Modularidad: {best_modularity:.4f}")
                print(f"  Tiempo de ejecución: {best_execution_time:.2f} segundos")

            except Exception as e:
                print(f"Error ejecutando {algo_name}: {str(e)}")
                # Añadir valores nulos para este algoritmo
                results['algorithm'].append(algo_name)
                results['num_communities'].append(None)
                results['modularity'].append(None)
                results['avg_community_size'].append(None)
                results['min_community_size'].append(None)
                results['max_community_size'].append(None)
                results['execution_time'].append(None)
                results['communities_distribution'].append(None)

        # Crear DataFrame y guardar resultados
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_file, index=False)
        print(f"\nResultados guardados en {output_file}")

        # Evaluar si hay estructura comunitaria clara
        valid_modularities = [m for m in results['modularity'] if m is not None]
        if valid_modularities:
            avg_modularity = sum(valid_modularities) / len(valid_modularities)
            if avg_modularity > 0.3:
                print("\nLa red del Fediverso muestra una estructura comunitaria clara.")
                print(f"Modularidad promedio: {avg_modularity:.4f}")
            else:
                print("\nLa red del Fediverso no muestra una estructura comunitaria fuerte.")
                print(f"Modularidad promedio: {avg_modularity:.4f}")

        # Identificar mejor algoritmo basado en modularidad
        if valid_modularities:
            best_algo_idx = results['modularity'].index(max(valid_modularities))
            print(f"\nMejor algoritmo: {results['algorithm'][best_algo_idx]}")
            print(f"Número de comunidades: {results['num_communities'][best_algo_idx]}")
            print(f"Modularidad: {results['modularity'][best_algo_idx]:.4f}")

        return results

    def visualize_community_results(self, community_file=None, results=None):
        """
        Visualiza los resultados de la detección de comunidades.

        Args:
            community_file (str): Ruta al archivo CSV con resultados (generado por detect_and_compare_communities)
            results (dict): Resultados directos del método detect_and_compare_communities

        Returns:
            matplotlib.figure.Figure: Figura con las gráficas generadas
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Cargar resultados si se proporciona un archivo
        if results is None and community_file is not None:
            df_results = pd.read_csv(community_file)
        elif results is not None:
            df_results = pd.DataFrame(results)
        else:
            print("Error: Debes proporcionar un archivo de resultados o un diccionario de resultados")
            return None

        # Configuración de la figura
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Comparación de Algoritmos de Detección de Comunidades en el Fediverso', fontsize=16)

        # Gráfica 1: Modularidad por algoritmo
        if not all(pd.isna(df_results['modularity'])):
            ax1 = axes[0, 0]
            sns.barplot(x='algorithm', y='modularity', data=df_results, ax=ax1)
            ax1.set_title('Modularidad por Algoritmo')
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
            ax1.set_ylim(0, max(df_results['modularity'].fillna(0)) * 1.1)

        # Gráfica 2: Número de comunidades por algoritmo
        if not all(pd.isna(df_results['num_communities'])):
            ax2 = axes[0, 1]
            sns.barplot(x='algorithm', y='num_communities', data=df_results, ax=ax2)
            ax2.set_title('Número de Comunidades por Algoritmo')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

        # Gráfica 3: Tiempo de ejecución por algoritmo
        if not all(pd.isna(df_results['execution_time'])):
            ax3 = axes[1, 0]
            sns.barplot(x='algorithm', y='execution_time', data=df_results, ax=ax3)
            ax3.set_title('Tiempo de Ejecución (segundos)')
            ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')

        # Gráfica 4: Tamaño promedio de comunidad por algoritmo
        if not all(pd.isna(df_results['avg_community_size'])):
            ax4 = axes[1, 1]
            sns.barplot(x='algorithm', y='avg_community_size', data=df_results, ax=ax4)
            ax4.set_title('Tamaño Promedio de Comunidad')
            ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        return fig

    def visualize_nested_circle_plot(self, output_file=None, num_shells=5, min_degree=1,
                                     width=1200, height=1200, show_labels=True, label_size=8,
                                     label_min_degree=10, edge_opacity=0.15, node_size_factor=30,
                                     colormap='tab20'):
        """
        Visualiza la estructura anidada (nestedness) del Fediverso usando un gráfico circular
        con nodos organizados en coronas/shells según su grado de conectividad

        Args:
            output_file (str, optional): Ruta para guardar la imagen. Si es None, solo muestra.
            num_shells (int): Número de coronas o shells para la visualización.
            min_degree (int): Grado mínimo de nodos a incluir.
            width (int): Ancho de la figura en píxeles.
            height (int): Altura de la figura en píxeles.
            show_labels (bool): Si se muestran las etiquetas de los nodos.
            label_size (int): Tamaño de fuente para las etiquetas.
            label_min_degree (int): Grado mínimo para que un nodo muestre su etiqueta.
            edge_opacity (float): Opacidad de las aristas (0-1).
            node_size_factor (float): Factor para escalar el tamaño de los nodos.
            colormap (str): Nombre del mapa de colores para software.

        Returns:
            self: Para encadenamiento de métodos
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import math
        from matplotlib.colors import Normalize

        print("Preparando visualización circular anidada...")

        # Filtrar nodos con grado mínimo
        if min_degree > 1:
            view = self.G.copy()
            nodes_to_remove = [node for node, degree in view.degree()
                               if degree < min_degree]
            view.remove_nodes_from(nodes_to_remove)
        else:
            view = self.G

        if len(view.nodes()) == 0:
            print("No hay nodos que cumplan el criterio de grado mínimo")
            return self

        # Obtener los grados de los nodos
        node_degrees = dict(view.degree())

        # Determinar rangos de grados para cada corona/shell
        min_deg = min(node_degrees.values())
        max_deg = max(node_degrees.values())

        # Calcular intervalos de grado para cada corona
        # Usamos escala logarítmica para que la distribución sea más uniforme
        if min_deg == max_deg:
            # Si todos los nodos tienen el mismo grado, usamos una sola corona
            shell_ranges = [(min_deg, max_deg)]
        else:
            if min_deg > 0:
                log_min = math.log(min_deg)
            else:
                log_min = 0
            log_max = math.log(max_deg)
            log_step = (log_max - log_min) / num_shells

            shell_ranges = []
            for i in range(num_shells):
                if i == num_shells - 1:
                    # Para la última corona, incluimos el valor máximo
                    shell_ranges.append((math.ceil(math.exp(log_min + i * log_step)),
                                         math.ceil(math.exp(log_min + (i + 1) * log_step))))
                else:
                    shell_ranges.append((math.ceil(math.exp(log_min + i * log_step)),
                                         math.floor(math.exp(log_min + (i + 1) * log_step))))

        # Distribuir nodos en las coronas según su grado
        shells = [[] for _ in range(len(shell_ranges))]
        for node, degree in node_degrees.items():
            for i, (min_shell_deg, max_shell_deg) in enumerate(shell_ranges):
                if min_shell_deg <= degree <= max_shell_deg:
                    shells[i].append(node)
                    break

        # Eliminar coronas vacías
        shells = [shell for shell in shells if shell]

        # Ordenar nodos dentro de cada corona para mejor visualización
        # Los ordenamos por software para que nodos del mismo tipo estén juntos
        for i, shell in enumerate(shells):
            shells[i] = sorted(shell, key=lambda x: view.nodes[x].get('software', ''))

        # Configurar colores basados en software
        software_types = set()
        for _, attrs in view.nodes(data=True):
            if 'software' in attrs and attrs['software']:
                software_types.add(attrs['software'])

        # Crear diccionario de colores para software
        cmap = plt.cm.get_cmap(colormap)
        color_dict = {software: cmap(i % cmap.N)
                      for i, software in enumerate(sorted(software_types))}
        color_dict[''] = (0.7, 0.7, 0.7, 1.0)  # Color por defecto para desconocido

        # Preparar figura
        plt.figure(figsize=(width / 100, height / 100), dpi=100)

        # Crear layout circular anidado
        pos = nx.shell_layout(view, shells)

        # Preparar tamaños y colores de nodos
        node_colors = []
        node_sizes = []
        for node in view.nodes():
            # Color según software
            software = view.nodes[node].get('software', '')
            node_colors.append(color_dict.get(software, color_dict['']))

            # Tamaño según grado (con un valor mínimo)
            size = node_size_factor * math.log1p(node_degrees[node])
            node_sizes.append(max(size, 5))

        # Dibujar aristas con baja opacidad
        nx.draw_networkx_edges(view, pos, alpha=edge_opacity, edge_color='gray', width=0.5)

        # Dibujar nodos
        nx.draw_networkx_nodes(view, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)

        # Añadir etiquetas para nodos con grado suficiente
        if show_labels:
            labels = {}
            for node in view.nodes():
                if node_degrees[node] >= label_min_degree:
                    labels[node] = node
            nx.draw_networkx_labels(view, pos, labels=labels, font_size=label_size,
                                    font_family='sans-serif')

        # Dibujar círculos concéntricos para indicar las coronas
        # Calculamos radios para los círculos basados en el layout
        center_x = sum(pos[node][0] for node in view.nodes()) / len(view.nodes())
        center_y = sum(pos[node][1] for node in view.nodes()) / len(view.nodes())

        # Encontrar el radio máximo desde el centro hasta algún nodo
        max_radius = max(math.sqrt((pos[node][0] - center_x) ** 2 +
                                   (pos[node][1] - center_y) ** 2)
                         for node in view.nodes())

        # Dibujar círculos concéntricos para las coronas
        for i in range(1, len(shells) + 1):
            radius = max_radius * i / len(shells)
            circle = plt.Circle((center_x, center_y), radius, fill=False,
                                color='gray', linestyle='--', alpha=0.5)
            plt.gca().add_patch(circle)

        # Anotar los rangos de grado en cada corona
        for i, (min_shell_deg, max_shell_deg) in enumerate(shell_ranges[:len(shells)]):
            radius = max_radius * (i + 0.5) / len(shells)
            angle = 45  # Ángulo en grados para la etiqueta
            x = center_x + radius * math.cos(math.radians(angle))
            y = center_y + radius * math.sin(math.radians(angle))

            if min_shell_deg == max_shell_deg:
                label = f"Grado: {min_shell_deg}"
            else:
                label = f"Grados: {min_shell_deg}-{max_shell_deg}"

            plt.annotate(label, xy=(x, y), xytext=(x, y), fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                         ha='center', va='center')

        # Crear leyenda para tipos de software
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=color_dict[software],
                                      markersize=8, label=software)
                           for software in sorted(color_dict.keys())]

        # Solo mostramos leyenda si no hay demasiados tipos de software
        if len(legend_elements) <= 15:
            plt.legend(handles=legend_elements, loc='upper right',
                       title='Software', fontsize='small')

        plt.axis('off')
        plt.title('Estructura Anidada del Fediverso', fontsize=14)

        # Añadir una anotación explicativa
        plt.annotate(
            f"Distribución en {len(shells)} coronas\nOrdenadas por grado de conectividad",
            xy=(0.02, 0.02), xycoords='axes fraction',
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
        )

        plt.tight_layout()

        # Guardar o mostrar
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            print(f"Visualización circular anidada guardada en: {output_file}")
        else:
            plt.show()

        return self

    def plot_betweenness_degree_correlation(self, metrics_file='centrality_metrics.csv',
                                            output_file='betweenness_degree_correlation.png'):
        """
        Genera un gráfico de dispersión entre la centralidad de intermediación y el grado de los nodos,
        incluyendo un ajuste de regresión lineal.

        Args:
            metrics_file (str): Ruta al archivo CSV con las métricas de centralidad
            output_file (str): Ruta donde guardar la imagen generada

        Returns:
            tuple: (figura, coeficientes de regresión, r-squared)
        """

        # Crear directorio de salida si no existe
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Cargar las métricas de centralidad
        print(f"Cargando métricas desde {metrics_file}...")
        df_metrics = pd.read_csv(metrics_file)

        # Verificar que tenemos las columnas necesarias
        required_cols = ['degree_centrality', 'betweenness_centrality']
        if not all(col in df_metrics.columns for col in required_cols):
            print(f"Error: El archivo de métricas debe contener las columnas {required_cols}")
            return None, None, None

        # Crear figura
        plt.figure(figsize=(12, 10))

        # Configurar estilo
        sns.set_style('whitegrid')

        # Crear gráfico de dispersión con regresión
        scatter = sns.regplot(
            x='degree_centrality',
            y='betweenness_centrality',
            data=df_metrics,
            scatter_kws={
                'alpha': 0.5,
                'color': 'red',
                's': 50  # Tamaño de los puntos
            }
        )

        # Añadir etiquetas y título
        plt.title('Correlación entre Centralidad de Intermediación y de Grado', fontsize=16)
        plt.xlabel('Centralidad de Grado', fontsize=14)
        plt.ylabel('Centralidad de Intermediación', fontsize=14)

        # Guardar figura
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado como: {output_file}")

        # Versión con escala logarítmica en ambos ejes
        plt.figure(figsize=(12, 10))

        # Filtrar valores > 0 para la escala logarítmica
        df_log = df_metrics[(df_metrics['degree_centrality'] > 0) & (df_metrics['betweenness_centrality'] > 0)]

        # Crear gráfico de dispersión con escala logarítmica
        scatter_log = sns.regplot(
            x='degree_centrality',
            y='betweenness_centrality',
            data=df_log,
            scatter_kws={
                'alpha': 0.5,
                'color': 'blue',
                's': 50
            },
        )

        # Convertir a escala logarítmica
        plt.xscale('log')
        plt.yscale('log')

        # Añadir etiquetas y título
        plt.title('Correlación entre Centralidad de Intermediación y de Grado (escala logarítmica)', fontsize=16)
        plt.xlabel('Centralidad de Grado (log)', fontsize=14)
        plt.ylabel('Centralidad de Intermediación (log)', fontsize=14)

        # Guardar figura con escala logarítmica
        log_output = output_file.replace('.png', '_log.png')
        plt.tight_layout()
        plt.savefig(log_output, dpi=300, bbox_inches='tight')
        print(f"Gráfico con escala logarítmica guardado como: {log_output}")

        # Crear histograma 2D
        plt.figure(figsize=(12, 10))
        heatmap = plt.hexbin(
            df_metrics['degree_centrality'],
            df_metrics['betweenness_centrality'],
            gridsize=50,
            cmap='viridis',
            mincnt=1,
            bins='log'  # Escala logarítmica para el conteo
        )

        # Añadir barra de color
        cbar = plt.colorbar(label='Conteo de nodos (log)')

        # Añadir etiquetas y título
        plt.title('Distribución de Densidad entre Centralidad de Intermediación y de Grado', fontsize=16)
        plt.xlabel('Centralidad de Grado', fontsize=14)
        plt.ylabel('Centralidad de Intermediación', fontsize=14)

        # Guardar histograma 2D
        heatmap_output = output_file.replace('.png', '_heatmap.png')
        plt.tight_layout()
        plt.savefig(heatmap_output, dpi=300, bbox_inches='tight')
        print(f"Histograma 2D guardado como: {heatmap_output}")

        return self

    def top_metrics(self, metrics_file='centrality_metrics.csv', top_n=20):
        df_metrics = pd.read_csv(metrics_file)
        metrics = [col for col in df_metrics.columns if col != 'node']

        for metric in metrics:
            print(f"\nAnalizando top {top_n} instancias por {metric}:")

            # Ordenar por la métrica actual (descendente) y tomar las top N
            top_df = df_metrics.sort_values(by=metric, ascending=False).head(top_n).copy()

            # Renombrar columnas para mejor presentación
            top_df = top_df[['node', metric]].rename(columns={'node': 'Instancia', metric: 'Valor'})

            # Mostrar resultados
            print(f"Top {top_n} instancias por {metric}:")
            print(top_df.to_string(index=False))

        return self

    def visualize_adjacency_matrix(self, output_file=None, top_instances=None,
                               colormap='viridis', width=1000, height=900, show_tick_labels=True,
                               tick_label_size=8, sort_by='degree'):
        """
        Visualiza la matriz de adyacencia del grafo ordenada por grados u otra métrica

        Args:
            output_file (str, optional): Ruta para guardar la imagen. Si es None, solo muestra.
            top_instances (int, optional): Limitar visualización a las N instancias con más conexiones.
            colormap (str): Nombre del mapa de colores de matplotlib a usar.
            width (int): Ancho de la figura en píxeles.
            height (int): Altura de la figura en píxeles.
            show_tick_labels (bool): Si se muestran las etiquetas de los ejes.
            tick_label_size (int): Tamaño de fuente para las etiquetas de los ejes.
            sort_by (str): Criterio para ordenar: 'degree' (default), 'in_degree', 'out_degree', 'users', 'software'.

        Returns:
            self: Para encadenamiento de métodos
        """
        from matplotlib.colors import LinearSegmentedColormap

        print("Preparando matriz de adyacencia...")

        # Obtener la lista de nodos
        if top_instances is not None:
            degrees = dict(self.G.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_instances]
            nodes = [node for node, _ in top_nodes]
        else:
            nodes = list(self.G.nodes())

        if len(nodes) == 0:
            print("No hay nodos para visualizar")
            return self

        # Ordenar nodos según el criterio especificado
        if sort_by == 'degree':
            nodes.sort(key=lambda x: self.G.degree(x), reverse=True)
        elif sort_by == 'in_degree' and isinstance(self.G, nx.DiGraph):
            nodes.sort(key=lambda x: self.G.in_degree(x), reverse=True)
        elif sort_by == 'out_degree' and isinstance(self.G, nx.DiGraph):
            nodes.sort(key=lambda x: self.G.out_degree(x), reverse=True)
        elif sort_by == 'users':
            nodes.sort(key=lambda x: self.G.nodes[x].get('users', 0)
            if isinstance(self.G.nodes[x].get('users', 0), (int, float)) else 0,
                       reverse=True)
        elif sort_by == 'software':
            nodes.sort(key=lambda x: self.G.nodes[x].get('software', ''))

        # Crear matriz de adyacencia ordenada
        adj_matrix = np.zeros((len(nodes), len(nodes)))
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        for i, source in enumerate(nodes):
            for target in self.G.neighbors(source):
                if target in node_to_idx:  # Asegurarse de que el destino está en nuestra lista filtrada
                    j = node_to_idx[target]
                    adj_matrix[i, j] = 1

        # Crear figura
        plt.figure(figsize=(width / 100, height / 100), dpi=100)

        # Personalizar el colormap para que el cero sea blanco o muy claro
        if colormap == 'custom':
            # Crear un colormap personalizado que va de blanco a azul oscuro
            colors = [(1, 1, 1), (0, 0, 0.8)]  # Blanco a azul oscuro
            custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
            cmap = custom_cmap
        else:
            cmap = plt.get_cmap(colormap)

        # Visualizar la matriz
        plt.imshow(adj_matrix, cmap=cmap, interpolation='none')
        plt.colorbar(label='Conexión')

        # Configurar etiquetas si se solicita
        if show_tick_labels:
            # Acortar nombres de dominio para que sean más legibles en las etiquetas
            shortened_labels = []
            for node in nodes:
                if len(node) > 15:  # Acortar dominios muy largos
                    parts = node.split('.')
                    if len(parts) > 2:
                        shortened = parts[0][:10] + "..." + '.'.join(parts[-2:])
                    else:
                        shortened = node[:15] + "..."
                else:
                    shortened = node
                shortened_labels.append(shortened)

            plt.xticks(range(len(nodes)), shortened_labels, rotation=90, fontsize=tick_label_size)
            plt.yticks(range(len(nodes)), shortened_labels, fontsize=tick_label_size)
        else:
            plt.xticks([])
            plt.yticks([])

        plt.title(f'Matriz de Adyacencia')
        plt.tight_layout()

        # Guardar o mostrar
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            print(f"Matriz de adyacencia guardada en: {output_file}")
        else:
            plt.show()

        return self

    def nodf(self):
        """
        Prepara la matriz de adyacencia ordenada por grado
        """
        # Obtener grados
        degree_dict = dict(self.G.degree())

        # Ordenar nodos por grado (descendente)
        sorted_nodes = sorted(degree_dict.keys(),
                                   key=lambda x: degree_dict[x],
                                   reverse=True)

        # Crear matriz de adyacencia ordenada
        adj_matrix = nx.adjacency_matrix(self.G, nodelist=sorted_nodes)
        adjacency_matrix = adj_matrix.toarray()

        n = len(sorted_nodes)
        matrix = adjacency_matrix

        # Calcular NODF para filas (N_rows)
        paired_overlap_rows = 0
        paired_rows = 0

        for i in range(n - 1):
            ki = np.sum(matrix[i, :])  # grado del nodo i
            for j in range(i + 1, n):
                kj = np.sum(matrix[j, :])  # grado del nodo j

                if ki > kj and kj > 0:  # Solo si ki > kj y kj > 0
                    # Calcular overlap
                    overlap = np.sum(matrix[i, :] * matrix[j, :])  # intersección
                    paired_overlap_rows += overlap / kj
                    paired_rows += 1

        # Calcular NODF para columnas (N_columns)
        paired_overlap_cols = 0
        paired_cols = 0

        for i in range(n - 1):
            ki = np.sum(matrix[:, i])  # grado de la columna i
            for j in range(i + 1, n):
                kj = np.sum(matrix[:, j])  # grado de la columna j

                if ki > kj and kj > 0:  # Solo si ki > kj y kj > 0
                    # Calcular overlap
                    overlap = np.sum(matrix[:, i] * matrix[:, j])  # intersección
                    paired_overlap_cols += overlap / kj
                    paired_cols += 1

        # NODF total
        if paired_rows + paired_cols > 0:
            nodf = 100 * (paired_overlap_rows + paired_overlap_cols) / (paired_rows + paired_cols)
        else:
            nodf = 0

        print(f'NODF (Nestedness based on Overlap and Decreasing Fill) = {nodf}')

        return nodf

nodes_file = "data/nodes_cleaned.csv"
edges_file = "fediverso_graph.json"

analyzer = FediverseAnalyzer(nodes_file, edges_file)
analyzer.load_instances()
analyzer.build_graph_from_json()
#analyzer.print_basic_stats()
#analyzer.calculate_and_save_centrality_metrics(output_file="centrality_metrics.csv")
#analyzer.top_metrics(metrics_file='centrality_metrics.csv', top_n=20)
#analyzer.detect_and_compare_communities(output_file='community_comparison.csv', num_runs=3)
#analyzer.plot_centrality_distributions(metrics_file='centrality_metrics.csv', output_dir='plots/')
#analyzer.plot_betweenness_degree_correlation(metrics_file='centrality_metrics.csv', output_file='plots/betweenness_degree_correlation.png')
#analyzer.visualize_adjacency_matrix(output_file="matriz_adyacencia.png", colormap='custom', sort_by='degree', show_tick_labels=False)
analyzer.nodf()