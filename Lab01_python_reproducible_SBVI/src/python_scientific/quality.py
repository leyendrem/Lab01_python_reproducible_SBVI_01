def validate_channel_configuration(
    observed_channels,
    required_channels,
):
    """Resume duplicados, faltantes y canales adicionales."""

    for channel in observed_channels:
        if not channel.strip():
            # .strip() elimina espacios en blanco y caracteres no deseados.
            raise ValueError("El nombre del canal no puede estar vacío.")

    for channel in required_channels:
        if not channel.strip():
            raise ValueError("El nombre del canal no puede estar vacío.")

    # Se crea un conjunto vacío, donde se irán registrando los elementos ya leídos.
    seen = set()
    # Se crea una lista, para agregar los canales que se repiten.
    duplicates = []

    for channel in observed_channels:
        if channel in seen:
            if channel not in duplicates:
                duplicates.append(channel)
        else:
            seen.add(channel)

    # Lista a conjunto para operaciones de los mismos (en memoria aparte)...
    # Operaciones como resta (-), unión (|) e intersección (&).
    observed_set = set(observed_channels)

    # Se organizan por abecedario para mejor visualización.
    missing = sorted(required_channels - observed_set)
    additional = sorted(observed_set - required_channels)

    # Se retorna un diccionario serializable...
    # Para ser serializable, valores deben poder ser convertidos a un formato de texto estándar ().
    # Conjuntos no son un tipo de dato compatible para el formato, por ejemplo, tipo JSON.
    return {
        "observed_channels": list(observed_channels),
        "required_channels": sorted(required_channels),
        "duplicates": duplicates,
        "missing": missing,
        "additional": additional,
    }
