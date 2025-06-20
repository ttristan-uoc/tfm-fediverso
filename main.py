import instances_list as ins
import geoloc
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
    geoloc.instances_map(fediverse_observer_df)
    geoloc.instances_heatmap(fediverse_observer_df)
    
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
    
    # Análisis macroscópico
    analyzer.analyze_macroscopic()
    analyzer.visualize_adjacency_matrix(output_file="adjacency_matrix.png", top_instances=None,
                                        colormap='viridis', width=1000, height=900, show_tick_labels=False,
                                        sort_by='degree')
    
    # Análisis microscópico: obtener métricas de centralidad y sus plots
    analyzer.analyze_microscopic()
    analyzer.calculate_and_save_centrality_metrics(output_file="centrality_metrics.csv")
    analyzer.top_metrics(metrics_file='centrality_metrics.csv', top_n=20)
    analyzer.plot_centrality_distributions(metrics_file='centrality_metrics.csv', output_dir='plots/')
    analyzer.plot_betweenness_degree_correlation(metrics_file='centrality_metrics.csv', output_file='plots/betweenness_degree_correlation.png')
    
    # Análisis mesoscópico: obtener y comparar comunidades
    analyzer.analyze_mesoscopic()
    analyzer.detect_and_compare_communities(output_file='community_comparison.csv', num_runs=3)
    analyzer.nodf()


