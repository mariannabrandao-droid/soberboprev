from __future__ import annotations

from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import folium


KML_NS = {"k": "http://www.opengis.net/kml/2.2"}
PONTE_NOME = "Ponte Sobre o Rio Soberbo"
POCO_NOME = "Poço Verde"

# Referência PRELIMINAR: não é estação existente e ainda não integra o modelo.
PC03_CANDIDATO = {
    "nome": "P3 candidato — trecho urbano",
    "lat": -22.507319,
    "lon": -42.987130,
}


def _parse_coord_text(texto: str):
    coords = []
    for token in (texto or "").strip().split():
        partes = token.split(",")
        if len(partes) >= 2:
            coords.append([float(partes[0]), float(partes[1])])
    return coords


def _ler_kml_raiz(caminho: Path):
    return ET.parse(caminho).getroot()


def _ler_kmz_raiz(caminho: Path):
    with zipfile.ZipFile(caminho, "r") as zf:
        nomes = [n for n in zf.namelist() if n.lower().endswith(".kml")]
        if not nomes:
            raise ValueError("Nenhum KML encontrado dentro do KMZ.")
        return ET.fromstring(zf.read(nomes[0]))


def _extrair_pontos(root):
    pontos = {}
    for pm in root.findall(".//k:Placemark", KML_NS):
        nome = (pm.findtext("k:name", default="", namespaces=KML_NS) or "").strip()
        pt = pm.find(".//k:Point", KML_NS)
        if pt is None:
            continue
        coords = _parse_coord_text(
            pt.findtext("k:coordinates", default="", namespaces=KML_NS)
        )
        if coords:
            lon, lat = coords[0]
            pontos[nome] = {"nome": nome, "lat": lat, "lon": lon}
    return pontos


def _extrair_linhas(root, nome_alvo):
    linhas = []
    for pm in root.findall(".//k:Placemark", KML_NS):
        nome = (pm.findtext("k:name", default="", namespaces=KML_NS) or "").strip()
        if nome.casefold() != nome_alvo.casefold():
            continue
        for ls in pm.findall(".//k:LineString", KML_NS):
            coords = _parse_coord_text(
                ls.findtext("k:coordinates", default="", namespaces=KML_NS)
            )
            if len(coords) >= 2:
                linhas.append(
                    {"nome": nome, "path": [[lat, lon] for lon, lat in coords]}
                )
    return linhas


def _extrair_poligonos(root):
    poligonos = []
    for pm in root.findall(".//k:Placemark", KML_NS):
        nome = (pm.findtext("k:name", default="", namespaces=KML_NS) or "").strip()
        for poly in pm.findall(".//k:Polygon", KML_NS):
            coords = _parse_coord_text(
                poly.findtext(
                    ".//k:outerBoundaryIs/k:LinearRing/k:coordinates",
                    default="",
                    namespaces=KML_NS,
                )
            )
            if len(coords) >= 3:
                poligonos.append(
                    {
                        "nome": nome or "Parque Nacional da Serra dos Órgãos",
                        "polygon": [[lat, lon] for lon, lat in coords],
                    }
                )
    return poligonos


def _selecionar_trecho_ponte_poco(linhas, ponte, poco):
    lat_min = min(ponte["lat"], poco["lat"]) - 0.0025
    lat_max = max(ponte["lat"], poco["lat"]) + 0.0025
    lon_min = min(ponte["lon"], poco["lon"]) - 0.0025
    lon_max = max(ponte["lon"], poco["lon"]) + 0.0025

    return [
        linha
        for linha in linhas
        if any(
            lat_min <= lat <= lat_max and lon_min <= lon <= lon_max
            for lat, lon in linha["path"]
        )
    ]


def carregar_camadas_mapa(caminho_projeto: Path, caminho_parnaso: Path):
    projeto = _ler_kml_raiz(caminho_projeto)
    parnaso_root = _ler_kmz_raiz(caminho_parnaso)

    pontos = _extrair_pontos(projeto)
    if PONTE_NOME not in pontos:
        raise ValueError(f"Ponto '{PONTE_NOME}' não encontrado.")
    if POCO_NOME not in pontos:
        raise ValueError(f"Ponto '{POCO_NOME}' não encontrado.")

    ponte = pontos[PONTE_NOME]
    poco = pontos[POCO_NOME]

    return {
        "pontos": pontos,
        "ponte": ponte,
        "poco": poco,
        "pc03": PC03_CANDIDATO,
        "rio": _extrair_linhas(projeto, "Rio Soberbo"),
        "trilha": _extrair_linhas(
            projeto, "Trilha - Pedra do Sino / Vale das Antes"
        ),
        "parnaso": _extrair_poligonos(parnaso_root),
    }


def criar_mapa_estudo(camadas):
    camadas["trecho"] = _selecionar_trecho_ponte_poco(
        camadas["rio"], camadas["ponte"], camadas["poco"]
    )

    mapa = folium.Map(
        location=[-22.5035, -43.0120],
        zoom_start=13,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri World Imagery",
        name="Satélite",
        overlay=False,
        control=True,
        max_zoom=19,
        show=True,
    ).add_to(mapa)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Cartográfico",
        overlay=False,
        control=True,
        show=False,
    ).add_to(mapa)

    # PARNASO
    fg_parnaso = folium.FeatureGroup(name="Limite PARNASO", show=True)
    for pol in camadas["parnaso"]:
        folium.PolyLine(
            pol["polygon"],
            color="#58D68D",
            weight=2,
            opacity=0.9,
            tooltip="Limite do PARNASO",
        ).add_to(fg_parnaso)
    fg_parnaso.add_to(mapa)

    # Trilha fica opcional
    fg_trilha = folium.FeatureGroup(name="Trilha do projeto", show=False)
    for linha in camadas["trilha"]:
        folium.PolyLine(
            linha["path"],
            color="#2E86C1",
            weight=2,
            opacity=0.5,
            tooltip="Trilha do projeto",
        ).add_to(fg_trilha)
    fg_trilha.add_to(mapa)

    # Rio completo
    fg_rio = folium.FeatureGroup(name="Rio Soberbo", show=True)
    for linha in camadas["rio"]:
        folium.PolyLine(
            linha["path"],
            color="#FF5A52",
            weight=3,
            opacity=0.9,
            tooltip="Rio Soberbo",
        ).add_to(fg_rio)
    fg_rio.add_to(mapa)

    # Trecho da V1
    fg_v1 = folium.FeatureGroup(name="Trecho V1: P1 → P2", show=True)
    for linha in camadas["trecho"]:
        folium.PolyLine(
            linha["path"],
            color="#FF1F1F",
            weight=6,
            opacity=1.0,
            tooltip="Trecho analisado na V1",
        ).add_to(fg_v1)
    fg_v1.add_to(mapa)

    folium.CircleMarker(
        [camadas["ponte"]["lat"], camadas["ponte"]["lon"]],
        radius=8, color="white", weight=2,
        fill=True, fill_color="#FF3B30", fill_opacity=1,
        tooltip="P1 — Ponte / sensor UFF",
        popup="<b>P1 — Ponte / sensor UFF</b><br>Ponto de monitoramento a montante.",
    ).add_to(mapa)

    folium.CircleMarker(
        [camadas["poco"]["lat"], camadas["poco"]["lon"]],
        radius=8, color="white", weight=2,
        fill=True, fill_color="#2389DA", fill_opacity=1,
        tooltip="P2 — Poço Verde",
        popup="<b>P2 — Poço Verde</b><br>Ponto intermediário de interesse.",
    ).add_to(mapa)

    folium.CircleMarker(
        [camadas["pc03"]["lat"], camadas["pc03"]["lon"]],
        radius=8, color="white", weight=2,
        fill=True, fill_color="#F5B041", fill_opacity=1,
        tooltip="P3 candidato — trecho urbano (em validação)",
        popup=(
            "<b>P3 candidato — trecho urbano</b><br>"
            "Referência espacial preliminar.<br>"
            "<b>Não é estação existente e ainda não integra a modelagem.</b>"
        ),
    ).add_to(mapa)

    # Região visual de edificações, baseada apenas na leitura do satélite.
    folium.Circle(
        [-22.5250, -42.9900],
        radius=1900,
        color="#F5B041",
        weight=2,
        opacity=0.8,
        fill=True,
        fill_color="#F5B041",
        fill_opacity=0.06,
        dash_array="8,8",
        tooltip=(
            "Ocupação/edificações — referência visual preliminar. "
            "Não representa população exposta nem mancha de inundação."
        ),
    ).add_to(mapa)

    folium.Marker(
        [-22.5265, -42.9890],
        icon=folium.DivIcon(
            html=(
                '<div style="font-size:13px;font-weight:700;color:white;'
                'text-shadow:-1px -1px 0 #000,1px -1px 0 #000,'
                '-1px 1px 0 #000,1px 1px 0 #000;white-space:nowrap;">'
                'Ocupação urbana / edificações a jusante</div>'
            )
        ),
    ).add_to(mapa)

    folium.LayerControl(collapsed=False, position="topright").add_to(mapa)
    return mapa
