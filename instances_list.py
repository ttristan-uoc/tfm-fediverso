import requests
import json
import pandas as pd


def get_instances_social_json():
    """
    Devuelve todas las instancias proporcionadas por instances.social en formato json
    """

    # API request
    url = "https://instances.social/api/1.0/instances/list"

    # Token
    headers = {
        "Authorization": "Bearer qY7KfUZ7lFrNQGDImQCfFHbdaPly4yXbX97IPCOlINJXY7BOoAdxA9zfl1xoGAzIFia3hSwLgOQbsQkICd4uoRa8UV1bwMWZBusx8UvN8UTvSc5Uj8BRuaZbgfpE34yy"
    }
    params = {
        "count": 0  # 0 para obtener todas las instancias
    }

    # HTTP GET request
    response = requests.get(url, headers=headers, params=params)

    # Response handling
    if response.status_code == 200:
        data = response.json()  # parsea los datos de la respuesta como json
        return data.get("instances", [])
    else:
        print(f"Error al obtener datos de instances.social: {response.status_code} - {response.text}")
        return None


def get_all_instances_from_instances_social():
    """
    Devuelve todas las instancias proporcionadas por instances.social en formato pandas DataFrame
    """

    # Captura de los datos de instances.social
    instances_social_json = get_instances_social_json()

    # Guardado de los datos
    with open("data/instances_social.json", "w", encoding="utf-8") as f:
        json.dump(instances_social_json, f, indent=4, ensure_ascii=False)
    print("Datos de instances.social guardados en data/instances_social.json")

    # Transformar de JSON a CSV
    instances_social_df = pd.read_json("data/instances_social.json")
    instances_social_df.to_csv("data/instances_social_df.csv", index=False)

    print("Datos de instances.social guardados en data/instances_social_df.csv")
    return instances_social_df


def fetch_fediverse_nodes():
    """
    Función para obtener todos los nodos del Fediverso
    """

    # API request
    url = "https://api.fediverse.observer/"

    query = """
    query {
      nodes {
        domain
        name
        softwarename
        fullversion
        country
        countryname
        city
        state
        lat
        long
        ipv6
        protocols
        services
        signup
        total_users
        active_users_halfyear
        active_users_monthly
        uptime_alltime
        metadescription
        metalocation
        owner
        date_created
        date_updated
        status
      }
    }
    """

    try:
        print("Enviando consulta para obtener todos los nodos...")
        response = requests.post(
            url,
            json={'query': query}
        )

        # Convertimos la respuesta a JSON
        data = response.json()

        # Verificamos errores
        if 'errors' in data:
            print('Error en la consulta GraphQL:', data['errors'])
            return None

        # Extraemos la información
        if 'data' in data and 'nodes' in data['data']:
            return data['data']['nodes']
        else:
            print('Estructura de respuesta inesperada:', data)
            return None
    except Exception as e:
        print(f'Error al realizar la consulta: {e}')
        return None


def save_nodes_to_file(nodes, filename='data/fediverse_observer.json'):
    """
    Función para guardar los datos en un archivo JSON
    """

    if not nodes:
        print("No hay datos para guardar.")
        return False

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(nodes, f, ensure_ascii=False, indent=2)

        print(f"Se han guardado {len(nodes)} instancias en '{filename}'")
        return True
    except Exception as e:
        print(f"Error al guardar el archivo: {e}")
        return False


def get_all_instances_from_fediverse_observer():
    """
    Devuelve todas las instancias proporcionadas por fediverse.observer en formato pandas DataFrame
    """

    nodes = fetch_fediverse_nodes()
    if not nodes:
        print("No se pudieron obtener datos. Finalizado.")

    filename = "data/fediverse_observer.json"
    save_nodes_to_file(nodes, filename)

    fediverse_observer_df = pd.read_json("data/fediverse_observer.json")
    fediverse_observer_df.to_csv("data/fediverse_observer_df.csv", index=False)

    print("Datos de instances.social guardados en data/fediverse_observer_df.csv")
    return fediverse_observer_df


def data_exploration(data):
    """
    Función que imprime los resultados de hacer una primera exploración de los datos
    """

    print("Info:\n", data.info())
    print("\nDescribe:\n", data.describe())
    print("\nShape:", data.shape)
    print("\nColumnas:", data.columns)
    print("\nTipos de datos:\n", data.dtypes)
    print("\nPrimeras filas:\n", data.head())
    print("\nValores nulos:\n", data.isnull().sum())


data_exploration(pd.read_csv('data/fediverse_observer_df.csv'))
data_exploration(pd.read_csv('data/instances_social_df.csv'))
data_exploration(pd.read_csv('data/nodes_cleaned.csv'))