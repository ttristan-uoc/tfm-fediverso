import pandas as pd
import requests
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
from urllib.parse import urlparse
import random

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='fediverso_crawler.log'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


compatible_softwares = [
    'mastodon',
    'wordpress',
    'peertube',
    'misskey',
    'pleroma',
    'sharkey',
    'akkoma',
    'friendica',
    'owncast',
    'hometown',
    'iceshrimp',
    'firefish',
    'cherrypick',
    'bookwyrm',
    'mitra'
]


# Cargar el dataset unificado
def load_dataset(file_path='source/data/combined_instances_data.csv'):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f"Archivo {file_path} no encontrado")
        exit(1)


# Función para obtener el dominio limpio
def clean_domain(domain):
    if pd.isna(domain):
        return None

    # Asegurar que sea string
    domain = str(domain).strip().lower()

    # Eliminar protocolo si existe
    if domain.startswith(('http://', 'https://')):
        parsed = urlparse(domain)
        domain = parsed.netloc

    # Eliminar posibles rutas o parámetros
    domain = domain.split('/')[0]

    return domain


# Determinar qué endpoint usar según el software
def get_peers_endpoint(domain, software):
    # Añadir https:// si no lo tiene
    if not domain.startswith(('http://', 'https://')):
        base_url = f"https://{domain}"
    else:
        base_url = domain

    # Endpoints específicos según el software
    software = software.lower() if software else ""

    if software == 'peertube':
        return f"{base_url}/api/v1/server/following"
    elif software == 'misskey':
        return f"{base_url}/api/federation/instances"
    else:
        # Para Mastodon y compatibles usar el endpoint genérico
        return f"{base_url}/api/v1/instance/peers"


# Procesar la respuesta según el software
def process_response(response, software):
    """
    Procesa la respuesta de la API según el software y devuelve una lista de dominios
    """
    try:
        data = response.json()

        software = software.lower() if software else ""

        if software == 'misskey':
            # Misskey devuelve una lista de objetos con campo 'host'
            peers = []
            for instance in data:
                if isinstance(instance, dict) and 'host' in instance:
                    peers.append(instance['host'])
            return peers
        elif software == 'peertube':
            # PeerTube devuelve una estructura específica
            if isinstance(data, dict) and 'data' in data:
                peers = [entry['following']['host'] for entry in data.get('data', []) if 'following' in entry and 'host' in entry['following']]
                return peers
            else:
                return data  # Si no tiene el formato esperado, devolver tal cual
        else:
            # Para Mastodon y otros, la respuesta debería ser una lista de dominios
            return data

    except Exception as e:
        logging.error(f"Error al procesar respuesta: {str(e)}")
        return []


# Función para eliminar duplicados y limpiar la lista de instancias federadas
def clean_peer_list(peers_list):
    """
    Elimina duplicados y asegura que los dominios estén en formato limpio
    """
    # Convertir a lista si es un tipo diferente
    if not isinstance(peers_list, list):
        if isinstance(peers_list, dict):
            peers_list = list(peers_list.keys())
        else:
            try:
                peers_list = list(peers_list)
            except:
                return []

    # Limpiar cada dominio y eliminar valores nulos o vacíos
    cleaned_peers = []
    for peer in peers_list:
        clean_peer = clean_domain(peer)
        if clean_peer:  # Solo añadir si no es None o vacío
            cleaned_peers.append(clean_peer)

    # Eliminar duplicados manteniendo el orden
    unique_peers = []
    seen = set()
    for peer in cleaned_peers:
        if peer not in seen:
            seen.add(peer)
            unique_peers.append(peer)

    return unique_peers


# Función para obtener las conexiones de una instancia
def fetch_connected_instances(instance_data):
    domain = instance_data['domain']
    software = instance_data['software']

    # Ignorar si el software no está en la lista de compatibles
    if software and software.lower() not in compatible_softwares:
        logging.info(f"Ignorando {domain} (software no compatible: {software})")
        return domain, []

    clean_domain_name = clean_domain(domain)
    if not clean_domain_name:
        return domain, []

    # Archivo de caché para esta instancia
    cache_file = f"cache/{clean_domain_name.replace('.', '_')}.json"

    # Si ya tenemos los datos en caché, devolverlos
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            logging.info(f"Datos de {domain} cargados desde caché")
            return domain, cached_data
        except Exception as e:
            logging.warning(f"Error al leer caché para {domain}: {str(e)}")

    try:
        # Obtener el endpoint apropiado
        endpoint = get_peers_endpoint(clean_domain_name, software)

        # Realizar la petición
        headers = {
            'User-Agent': 'FediversoStudyBot/1.0 (Estudio académico sobre el Fediverso)'
        }
        response = requests.get(endpoint, headers=headers, timeout=15)

        if response.status_code == 200:
            try:
                # Procesar la respuesta según el software
                peers_list = process_response(response, software)

                # Limpiar y eliminar duplicados
                unique_peers = clean_peer_list(peers_list)

                # Guardar en caché la lista limpia de dominios
                os.makedirs('cache', exist_ok=True)
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(unique_peers, f)

                initial_count = len(peers_list) if isinstance(peers_list, list) else 0
                final_count = len(unique_peers)

                if initial_count != final_count:
                    logging.info(
                        f"Datos de {domain}: {initial_count} conexiones reducidas a {final_count} (eliminados {initial_count - final_count} duplicados)")
                else:
                    logging.info(f"Obtenidos datos de {domain} con {len(unique_peers)} conexiones únicas")

                return domain, unique_peers
            except json.JSONDecodeError:
                logging.error(f"Error al decodificar JSON para {domain}")
                return domain, []
        else:
            logging.warning(f"Error {response.status_code} al acceder a {endpoint}")
            return domain, []

    except requests.RequestException as e:
        logging.error(f"Error de conexión para {domain}: {str(e)}")
        return domain, []
    except Exception as e:
        logging.error(f"Error inesperado para {domain}: {str(e)}")
        return domain, []


# Función para procesar un lote de instancias
def process_batch(instances_batch):
    connections = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_connected_instances, instances_batch))

    for domain, peers in results:
        if domain and peers:
            connections[domain] = peers

    return connections


# Función principal
def build_fediverso_graph(dataset_path='fediverso_instancias_unificado.csv', output_path='fediverso_graph.json'):
    # Crear directorio de caché si no existe
    os.makedirs('cache', exist_ok=True)

    # Cargar el dataset
    logging.info("Cargando dataset...")
    df = load_dataset(dataset_path)

    # Limpiar dominios
    df['domain'] = df['domain'].apply(clean_domain)
    df = df.dropna(subset=['domain']).drop_duplicates(subset=['domain'])

    # Filtrar por softwares compatibles
    compatible_df = df[df['software'].str.lower().isin(compatible_softwares) | df['software'].isna()]
    logging.info(f"Dataset filtrado: {len(compatible_df)} instancias compatibles de {len(df)} totales")

    # Convertir DataFrame a lista de diccionarios
    instances = compatible_df.to_dict('records')
    logging.info(f"Analizando {len(instances)} instancias...")

    # Mezclar aleatoriamente las instancias para distribuir la carga
    random.shuffle(instances)

    # Procesar en lotes para evitar saturar la memoria
    batch_size = 100
    all_connections = {}

    for i in tqdm(range(0, len(instances), batch_size)):
        batch = instances[i:i + batch_size]
        batch_connections = process_batch(batch)
        all_connections.update(batch_connections)

        # Guardar resultados parciales cada cierto número de lotes
        if i % 500 == 0:
            with open(f"{output_path}.partial", 'w', encoding='utf-8') as f:
                json.dump(all_connections, f)
            logging.info(f"Guardado parcial: {len(all_connections)} instancias procesadas")

    # Guardar resultados finales
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_connections, f)

    logging.info(f"Proceso completado. Datos guardados en {output_path}")

    return all_connections
