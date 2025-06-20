import pandas as pd


def data_combination(instances_social_df, fediverse_observer_df):
    """
    Función para unir los datasets en los campos en común
    :param instances_social_df: dataset con los datos obtenidos de instances_social en formato pandas DataFrame
    :param fediverse_observer_df: dataset con los datos obtenidos de fediverse_observer en formato pandas DataFrame
    :return: pandas DataFrame
    """

    # Seleccionar y renombrar columnas de fediverse_observer_df para consistencia
    fo_df = fediverse_observer_df[['domain', 'softwarename', 'total_users', 'signup', 'uptime_alltime']].copy()
    fo_df.rename(columns={
        'softwarename': 'software',
        'total_users': 'users',
    }, inplace=True)

    # Seleccionar y renombrar columnas de instances_social_df para consistencia
    is_df = instances_social_df[['name', 'users', 'open_registrations']].copy()
    is_df.rename(columns={
        'name': 'domain',
        'open_registrations': 'signup'
    }, inplace=True)

    is_df['software'] = 'mastodon'

    # Combinar datos
    combined_df = pd.concat([fo_df, is_df], ignore_index=True)
    combined_df = combined_df.sort_values(by=['domain'])
    combined_df = combined_df.drop_duplicates(subset=['domain'], keep='first')

    # Guardar datos combinados como csv
    combined_df.to_csv('data/combined_instances_data.csv', index=False)

    return combined_df


def stats(combined_data):
    """
    Función que imprime estadísticas relevantes sobre los datos combinados
    :param combined_data: pandas DataFrame
    :return: None
    """

    print(f"Total de instancias unificadas: {len(combined_data)}")
    print(f"Distribución por software:")
    print(combined_data['software'].value_counts())
    print(f"Instancias con registros abiertos: {combined_data['signup'].sum()}")