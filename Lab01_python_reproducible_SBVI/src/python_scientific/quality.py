def validate_channel_configuration(
    observed_channels: list[str],
    required_channels: set[str],
) -> dict[str, object]:
    """Resume duplicados, faltantes y canales adicionales."""

    for channel in observed_channels:
        if not channel.strip():
            # .strip() elimina espacios en blanco y caracteres no deseados.
            raise ValueError("El nombre del canal no puede estar vacío.")

    for channel in required_channels:
        if not channel.strip():
            raise ValueError("El nombre del canal no puede estar vacío.")

    seen = set()
    duplicates = []

    for channel in observed_channels:
        if channel in seen:
            if channel not in duplicates:
                duplicates.append(channel)
        else:
            seen.add(channel)

    observed_set = set(observed_channels)

    missing = sorted(required_channels - observed_set)
    additional = sorted(observed_set - required_channels)

    # Conjuntos no son un tipo de dato compatible para el formato, por ejemplo, tipo JSON.
    return {
        "observed_channels": list(observed_channels),
        "required_channels": sorted(required_channels),
        "duplicates": duplicates,
        "missing": missing,
        "additional": additional,
    }
