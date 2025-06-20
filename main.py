import instances_list as ins
from geoloc import instances_map
import combination as comb
import argparse
import connections as cons
from analysis import FediverseAnalyzer


if __name__ == "__main__":
    # Obtención de datos de la API de instances.social
    instances_social_df = ins.get_all_instances_from_instances_social()

    # Obtención de datos de la API de fediverse.observer
    fediverse_observer_df = ins.get_all_instances_from_fediverse_observer()

    # Exploración inicial de los datasets
    ins.data_exploration(instances_social_df)
    ins.data_exploration(fediverse_observer_df)

    # Geolocalización de las instancias
    instances_map(fediverse_observer_df)

    # Unión de los datasets
    combined_data = comb.data_combination(instances_social_df, fediverse_observer_df)
    comb.stats(combined_data)

    # Configuración de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Crawl Fediverse instances and build connection graph')
    parser.add_argument('--input', default='data/combined_instances_data.csv',
                        help='Path to the unified instances dataset')
    parser.add_argument('--output', default='fediverso_graph.json', help='Path to save the output graph')
    args = parser.parse_args()

    # Web Crawler para obtener aristas
    connections = cons.build_fediverso_graph(args.input, args.output)

    # Limpiamos nodos
    nodes_file = "data/combined_instances_data.csv"
    edges_file = "fediverso_graph.json"

    cleaner = FediverseAnalyzer(nodes_file, edges_file)
    cleaner.clean_nodes()
    cleaner.export_cleaned_nodes_to_csv(output_file="data/nodes_cleaned.csv")

    # Cargamos analyzer con los nodos limpios
    nodes_file_clean = "data/nodes_cleaned.csv"

    # Realizamos los análisis sobre el dataset limpio
    analyzer = FediverseAnalyzer(nodes_file_clean, edges_file)
    analyzer.load_instances()           # Cargamos las instancias
    analyzer.build_graph_from_json()    # Construimos el grafo a partir de las aristas
    analyzer.print_basic_stats()        # Estadísticas básicas
    analyzer.analyze_microscopic()      # Análisis microscópico
    analyzer.analyze_mesoscopic()       # Análisis mesoscópico
    analyzer.analyze_macroscopic()      # Análisis macroscópico

    # Análisis extra
    # Matriz de adyacencia
    analyzer.visualize_adjacency_matrix(output_file="adjacency_matrix.png", top_instances=None,
                                        colormap='viridis', width=1000, height=900, show_tick_labels=False,
                                        sort_by='degree')
    # Correlación intermediación-grado
    analyzer.plot_betweenness_degree_correlation(output_file="betweenness_vs_degree.png",
                                                 width=900, height=700, annotate_top=10,
                                                 colormap='viridis',
                                                 highlight_software=None, log_scale=True)



