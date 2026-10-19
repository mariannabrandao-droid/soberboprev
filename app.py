from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from config import PONTE, POCO_VERDE, DISTANCIA_RETA_KM
from src.sensor_thingspeak import painel_sensor
from src.demo_soberbo import pagina_resumo_demo, pagina_sensor_demo, pagina_muskingum_demo, parametros
from src.ana import carregar_chuva_ana, resumo_chuva
from src.rating_curve import nivel_para_vazao
from src.muskingum import rotear_muskingum
from src.synthetic import gerar_niveis_sinteticos
from src.geo import carregar_camadas_mapa, criar_mapa_estudo

from src.weather import (
    carregar_previsao_open_meteo,
    resumo_previsao,
    preparar_previsao_diaria_interface,
    filtrar_previsao_futura,
)

from src.operational_forecast import (
    prever_cota_Dmais1,
)


# ================================================================
# CONFIGURAÇÃO
# ================================================================

st.set_page_config(
    page_title="SoberboPrev",
    page_icon="🌊",
    layout="wide",
)


@st.cache_data(ttl=1800)
def obter_previsao_meteorologica():
    """
    Consulta a Open-Meteo e mantém o resultado
    em cache por 30 minutos.
    """

    return carregar_previsao_open_meteo(
        latitude=PONTE["lat"],
        longitude=PONTE["lon"],
    )


# IDENTIDADE VISUAL — inspiração institucional, sem uso de marca oficial
st.markdown("""<style>
.stApp {background:#f7f9fc;color:#17324f}
[data-testid="stSidebar"] {background:#e9eff7}
h1,h2,h3 {color:#104b83;font-family:Arial,sans-serif}
[data-testid="stMetric"] {background:#ffffff;border:1px solid #dce5ef;border-radius:10px;padding:14px}
.stTabs [data-baseweb="tab"] {font-weight:600}
.stButton>button {border-radius:7px}
</style>""",unsafe_allow_html=True)
st.title('SoberboPrev | Rio Soberbo')
st.caption('Monitoramento hidrológico • Guapimirim/RJ • Plataforma experimental de apoio à Defesa Civil')
with st.sidebar:
    st.header('SoberboPrev')
    st.caption('Navegação e transparência dos dados')
    st.info('Monitoramento por sensor ThingSpeak')
    st.info('Previsão meteorológica Open-Meteo')
    st.warning('Modelos hidrológicos em validação')
    st.caption('Não constitui sistema oficial de alerta.')
st.session_state['v4_parametros'] = parametros()
abas = st.tabs([
    'Visão geral',
    'Sensor e leituras',
    'Previsão D+1',
    'Previsão meteorológica',
    'Dados e modelo',
    'Propagação Muskingum',
    'Engenharia e validação',
])
with abas[0]:
    pagina_resumo_demo()
with abas[1]:
    pagina_sensor_demo()

# ================================================================
# 1. PAINEL OPERACIONAL
# ================================================================

with abas[2]:

    st.subheader(
        "Previsão experimental — Rio Soberbo"
    )

    st.caption(
        "Estimativa da cota para D+1 a partir da "
        "condição atual do rio, precipitação antecedente "
        "e precipitação futura prevista."
    )

    # ------------------------------------------------------------
    # PREVISÃO METEOROLÓGICA
    # ------------------------------------------------------------

    try:

        previsao_met = (
            obter_previsao_meteorologica()
        )

        diario_operacional = (
            preparar_previsao_diaria_interface(
                previsao_met
            )
        )

        linha_d1 = diario_operacional[
            diario_operacional["horizonte"]
            == "D+1"
        ]

        if linha_d1.empty:

            chuva_d1_openmeteo = None
            data_d1 = None

        else:

            chuva_d1_openmeteo = float(
                linha_d1.iloc[0][
                    "precipitacao_mm"
                ]
            )

            data_d1 = (
                linha_d1.iloc[0]["data"]
            )

    except Exception:

        chuva_d1_openmeteo = None
        data_d1 = None

    # ------------------------------------------------------------
    # ENTRADAS TEMPORÁRIAS
    # ------------------------------------------------------------

    with st.expander(
        "⚙️ Dados hidrológicos atuais — entrada temporária",
        expanded=True,
    ):

        st.info(
            "Enquanto o sensor de nível e uma fonte "
            "observacional recente de precipitação não "
            "estiverem integrados, estes dados devem ser "
            "informados manualmente."
        )

        col_input_1, col_input_2 = st.columns(2)

        with col_input_1:

            cota_atual = st.number_input(
                "Cota atual",
                min_value=0.0,
                value=60.0,
                step=1.0,
                help=(
                    "Utilize a mesma unidade e referência "
                    "da série histórica usada no modelo. "
                    "Nesta etapa, a entrada é manual."
                ),
            )

            chuva_hoje = st.number_input(
                "Chuva observada hoje (mm)",
                min_value=0.0,
                value=0.0,
                step=0.1,
            )

            chuva_dmenos1 = st.number_input(
                "Chuva observada ontem — D-1 (mm)",
                min_value=0.0,
                value=0.0,
                step=0.1,
            )

            chuva_dmenos2 = st.number_input(
                "Chuva observada — D-2 (mm)",
                min_value=0.0,
                value=0.0,
                step=0.1,
            )

        with col_input_2:

            chuva_dmenos3 = st.number_input(
                "Chuva observada — D-3 (mm)",
                min_value=0.0,
                value=0.0,
                step=0.1,
            )

            chuva_dmenos4 = st.number_input(
                "Chuva observada — D-4 (mm)",
                min_value=0.0,
                value=0.0,
                step=0.1,
            )

            chuva_dmenos5 = st.number_input(
                "Chuva observada — D-5 (mm)",
                min_value=0.0,
                value=0.0,
                step=0.1,
            )

            chuva_dmenos6 = st.number_input(
                "Chuva observada — D-6 (mm)",
                min_value=0.0,
                value=0.0,
                step=0.1,
            )

        if chuva_d1_openmeteo is not None:

            st.success(
                "Precipitação prevista para D+1 carregada "
                f"automaticamente: {chuva_d1_openmeteo:.1f} mm"
            )

        else:

            st.error(
                "Não foi possível carregar automaticamente "
                "a precipitação prevista para D+1."
            )

    # ------------------------------------------------------------
    # BOTÃO
    # ------------------------------------------------------------

    gerar_previsao = st.button(
        "Gerar previsão D+1",
        type="primary",
        width="stretch",
    )

    if gerar_previsao:

        if chuva_d1_openmeteo is None:

            st.error(
                "A previsão não pode ser calculada porque "
                "a precipitação D+1 não está disponível."
            )

        else:

            try:

                resultado_prev = prever_cota_Dmais1(
                    cota_atual=cota_atual,
                    chuva_hoje=chuva_hoje,
                    chuva_d1=chuva_dmenos1,
                    chuva_d2=chuva_dmenos2,
                    chuva_d3=chuva_dmenos3,
                    chuva_d4=chuva_dmenos4,
                    chuva_d5=chuva_dmenos5,
                    chuva_d6=chuva_dmenos6,
                    chuva_prevista_amanha=
                        chuva_d1_openmeteo,
                )

                st.divider()

                st.markdown(
                    "### Resultado da previsão"
                )

                r1, r2, r3, r4 = st.columns(4)

                r1.metric(
                    "Cota atual",
                    f"{resultado_prev['cota_atual']:.1f}",
                    help=(
                        "Entrada manual nesta versão. "
                        "Futuramente poderá vir do sensor."
                    ),
                )

                r2.metric(
                    "Chuva prevista D+1",
                    (
                        f"{resultado_prev['chuva_prevista_Dmais1']:.1f} mm"
                    ),
                    help="Fonte: Open-Meteo.",
                )

                r3.metric(
                    "Cota prevista D+1",
                    (
                        f"{resultado_prev['cota_prevista_Dmais1']:.1f}"
                    ),
                )

                r4.metric(
                    "Variação prevista",
                    (
                        f"{resultado_prev['variacao_prevista']:+.1f}"
                    ),
                )

                # ------------------------------------------------
                # CONDIÇÃO
                # ------------------------------------------------

                cota_prev = (
                    resultado_prev[
                        "cota_prevista_Dmais1"
                    ]
                )

                p90 = (
                    resultado_prev[
                        "referencia_P90"
                    ]
                )

                p95 = (
                    resultado_prev[
                        "referencia_P95"
                    ]
                )

                if cota_prev >= p95:

                    st.warning(
                        "A cota prevista encontra-se acima "
                        "do P95 histórico da série utilizada."
                    )

                elif cota_prev >= p90:

                    st.warning(
                        "A cota prevista encontra-se acima "
                        "do P90 histórico da série utilizada."
                    )

                else:

                    st.success(
                        "A cota prevista permanece abaixo "
                        "do P90 histórico da série utilizada."
                    )

                st.caption(
                    "P90 e P95 são referências estatísticas "
                    "exploratórias da série histórica. "
                    "Não representam cotas oficiais de alerta, "
                    "transbordamento ou inundação."
                )

                # ------------------------------------------------
                # GRÁFICO COTA ATUAL → D+1
                # ------------------------------------------------

                st.markdown(
                    "### Trajetória prevista"
                )

                fig_cota = go.Figure()

                fig_cota.add_trace(
                    go.Scatter(
                        x=[
                            "Hoje",
                            "D+1",
                        ],
                        y=[
                            resultado_prev[
                                "cota_atual"
                            ],
                            resultado_prev[
                                "cota_prevista_Dmais1"
                            ],
                        ],
                        mode="lines+markers",
                        name="Cota",
                    )
                )

                fig_cota.add_hline(
                    y=p90,
                    line_dash="dash",
                    annotation_text=(
                        f"P90 histórico = {p90:.1f}"
                    ),
                )

                fig_cota.add_hline(
                    y=p95,
                    line_dash="dot",
                    annotation_text=(
                        f"P95 histórico = {p95:.1f}"
                    ),
                )

                fig_cota.update_layout(
                    xaxis_title="Horizonte",
                    yaxis_title="Cota",
                    hovermode="x unified",
                    margin=dict(
                        l=20,
                        r=20,
                        t=30,
                        b=20,
                    ),
                )

                st.plotly_chart(
                    fig_cota,
                    width="stretch",
                )

                # ------------------------------------------------
                # LEITURA DO RESULTADO
                # ------------------------------------------------

                st.markdown(
                    "### Leitura do resultado"
                )

                if (
                    resultado_prev[
                        "variacao_prevista"
                    ] > 0
                ):

                    st.write(
                        "O modelo estima elevação da cota "
                        "entre a condição atual informada "
                        "e D+1."
                    )

                elif (
                    resultado_prev[
                        "variacao_prevista"
                    ] < 0
                ):

                    st.write(
                        "O modelo estima redução da cota "
                        "entre a condição atual informada "
                        "e D+1."
                    )

                else:

                    st.write(
                        "O modelo estima estabilidade "
                        "aproximada da cota em D+1."
                    )

                st.info(
                    "A previsão D+1 utiliza um modelo de "
                    "regressão desenvolvido a partir da série "
                    "histórica ANA. A precipitação futura é "
                    "obtida da Open-Meteo. A cota atual e a "
                    "precipitação antecedente ainda dependem "
                    "de entrada manual."
                )

            except Exception as exc:

                st.error(
                    "Não foi possível gerar a previsão."
                )

                st.exception(exc)

    # ------------------------------------------------------------
    # MAPA
    # ------------------------------------------------------------

    st.divider()

    st.markdown(
        "### Área de estudo"
    )

    caminho_projeto = (
        Path(__file__).parent
        / "data"
        / "Projeto_Rio_Soberbo.kml"
    )

    caminho_parnaso = (
        Path(__file__).parent
        / "data"
        / "PN_Serra_Orgaos.kmz"
    )

    try:

        camadas_geo = carregar_camadas_mapa(
            caminho_projeto,
            caminho_parnaso,
        )

        mapa = criar_mapa_estudo(
            camadas_geo
        )

        components.html(
            mapa.get_root().render(),
            height=600,
            scrolling=False,
        )

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "P1",
            "Ponte / sensor"
        )

        m2.metric(
            "P2",
            "Poço Verde"
        )

        m3.metric(
            "P3",
            "Ponto candidato"
        )

        st.caption(
            "Os pontos representam referências do projeto. "
            "A área destacada no mapa não constitui "
            "mancha de inundação."
        )

    except Exception as exc:

        st.warning(
            "Não foi possível carregar o mapa."
        )

        with st.expander(
            "Ver detalhe do erro"
        ):

            st.exception(exc)


# ================================================================
# 2. PREVISÃO METEOROLÓGICA
# ================================================================

with abas[3]:

    st.subheader(
        "Previsão meteorológica"
    )

    st.caption(
        "Precipitação prevista para o ponto de "
        "referência da Ponte sobre o Rio Soberbo."
    )

    st.info(
        "Fonte: Open-Meteo. A previsão de precipitação "
        "constitui uma entrada do modelo hidrológico. "
        "Probabilidade de chuva não equivale a "
        "probabilidade de inundação."
    )

    try:

        previsao = (
            obter_previsao_meteorologica()
        )

        previsao_futura = (
            filtrar_previsao_futura(
                previsao
            )
        )

        resumo_met = resumo_previsao(
            previsao
        )

        diario = (
            preparar_previsao_diaria_interface(
                previsao
            )
        )

        if previsao_futura.empty:

            st.warning(
                "Não há previsão futura disponível."
            )

        else:

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Próximas 24 h",
                (
                    f"{resumo_met['chuva_24h_mm']:.1f} mm"
                ),
            )

            c2.metric(
                "Próximas 72 h",
                (
                    f"{resumo_met['chuva_72h_mm']:.1f} mm"
                ),
            )

            c3.metric(
                "Maior chuva horária",
                (
                    f"{resumo_met['max_chuva_horaria_mm']:.1f} mm/h"
                ),
            )

            c4.metric(
                "Maior probabilidade",
                (
                    f"{resumo_met['max_probabilidade_pct']:.0f}%"
                ),
            )

            st.markdown(
                "### Evolução horária da precipitação"
            )

            fig_prev = go.Figure()

            fig_prev.add_bar(
                x=previsao_futura[
                    "data_hora"
                ],
                y=previsao_futura[
                    "precipitacao_mm"
                ],
                name="Precipitação prevista",
            )

            fig_prev.update_layout(
                xaxis_title="Data e hora",
                yaxis_title="Precipitação (mm/h)",
                hovermode="x unified",
                margin=dict(
                    l=20,
                    r=20,
                    t=30,
                    b=20,
                ),
            )

            st.plotly_chart(
                fig_prev,
                width="stretch",
            )

            with st.expander(
                "Ver probabilidade horária"
            ):

                fig_prob = go.Figure()

                fig_prob.add_trace(
                    go.Scatter(
                        x=previsao_futura[
                            "data_hora"
                        ],
                        y=previsao_futura[
                            "probabilidade_precipitacao_pct"
                        ],
                        mode="lines",
                        name=(
                            "Probabilidade de precipitação"
                        ),
                    )
                )

                fig_prob.update_layout(
                    xaxis_title="Data e hora",
                    yaxis_title="Probabilidade (%)",
                    yaxis=dict(
                        range=[0, 100]
                    ),
                    hovermode="x unified",
                )

                st.plotly_chart(
                    fig_prob,
                    width="stretch",
                )

            st.markdown(
                "### Horizonte diário"
            )

            if not diario.empty:

                fig_diario = go.Figure()

                fig_diario.add_bar(
                    x=diario[
                        "horizonte"
                    ],
                    y=diario[
                        "precipitacao_mm"
                    ],
                    text=diario[
                        "precipitacao_mm"
                    ].map(
                        lambda x: (
                            f"{x:.1f} mm"
                        )
                    ),
                    textposition="outside",
                    name="Chuva prevista",
                )

                fig_diario.update_layout(
                    xaxis_title="Horizonte",
                    yaxis_title=(
                        "Precipitação prevista (mm)"
                    ),
                )

                st.plotly_chart(
                    fig_diario,
                    width="stretch",
                )

                tabela_diaria = (
                    diario.copy()
                )

                tabela_diaria[
                    "data"
                ] = pd.to_datetime(
                    tabela_diaria["data"]
                ).dt.strftime(
                    "%d/%m/%Y"
                )

                tabela_diaria = (
                    tabela_diaria.rename(
                        columns={
                            "horizonte":
                                "Horizonte",
                            "data":
                                "Data",
                            "precipitacao_mm":
                                "Precipitação prevista (mm)",
                            "probabilidade_max_pct":
                                "Probabilidade máxima (%)",
                        }
                    )
                )

                st.dataframe(
                    tabela_diaria,
                    width="stretch",
                    hide_index=True,
                )

            with st.expander(
                "Ver dados horários utilizados"
            ):

                tabela_horaria = (
                    previsao_futura.copy()
                )

                tabela_horaria[
                    "data_hora"
                ] = tabela_horaria[
                    "data_hora"
                ].dt.strftime(
                    "%d/%m/%Y %H:%M"
                )

                st.dataframe(
                    tabela_horaria,
                    width="stretch",
                    hide_index=True,
                )


    except Exception as exc:

        st.error(
            "Não foi possível consultar "
            "a previsão meteorológica."
        )

        st.exception(exc)


# ================================================================
# 3. DADOS E MODELO
# ================================================================

with abas[4]:

    st.subheader(
        "Dados históricos e modelo preditivo"
    )

    st.markdown(
        """
O modelo experimental D+1 foi desenvolvido para estimar
a **cota futura do Rio Soberbo**, utilizando a condição
hidrológica antecedente e a precipitação.

A estrutura atualmente utilizada é:

**cota atual + chuva atual + chuva antecedente +
chuva prevista D+1 → cota prevista D+1**
        """
    )

    # ------------------------------------------------------------
    # MODELO
    # ------------------------------------------------------------

    st.markdown(
        "### Modelo atualmente utilizado"
    )

    mod1, mod2, mod3, mod4 = (
        st.columns(4)
    )

    mod1.metric(
        "Modelo",
        "Regressão linear"
    )

    mod2.metric(
        "Horizonte",
        "D+1"
    )

    mod3.metric(
        "RMSE teste",
        "10,60"
    )

    mod4.metric(
        "R² teste",
        "0,791"
    )

    st.caption(
        "Resultados obtidos na avaliação cronológica "
        "realizada durante o desenvolvimento do modelo."
    )

    st.markdown(
        """
Na avaliação geral do conjunto de teste, a inclusão da
precipitação de D+1 reduziu o RMSE de aproximadamente
**11,66 para 10,60** e elevou o R² de aproximadamente
**0,747 para 0,791** em relação à regressão sem chuva futura.

Nos eventos de cota elevada, entretanto, o desempenho
permanece mais limitado. A inclusão da chuva futura
melhorou a estimativa da magnitude das cotas elevadas,
mas não resolveu todos os eventos de elevação abrupta.

Por esse motivo, o modelo deve ser interpretado como
**experimental**, e não como sistema validado de alerta.
        """
    )

    st.markdown(
        "### Variáveis utilizadas"
    )

    variaveis_modelo = pd.DataFrame(
        [
            ["Cota atual", "Condição do rio"],
            ["Chuva do dia", "Precipitação antecedente"],
            ["Chuva D-1", "Precipitação antecedente"],
            ["Chuva D-2", "Precipitação antecedente"],
            ["Chuva D-3", "Precipitação antecedente"],
            ["Acumulado 2 dias", "Precipitação antecedente"],
            ["Acumulado 3 dias", "Precipitação antecedente"],
            ["Acumulado 5 dias", "Precipitação antecedente"],
            ["Acumulado 7 dias", "Precipitação antecedente"],
            [
                "Chuva prevista D+1",
                "Forçante meteorológica futura",
            ],
        ],
        columns=[
            "Variável",
            "Função no modelo",
        ],
    )

    st.dataframe(
        variaveis_modelo,
        width="stretch",
        hide_index=True,
    )

    # ------------------------------------------------------------
    # HISTÓRICO ANA
    # ------------------------------------------------------------

    st.markdown(
        "### Série histórica de precipitação — ANA"
    )

    caminho_ana = (
        Path(__file__).parent
        / "data"
        / "2242094_Chuvas.csv"
    )

    try:

        chuva = carregar_chuva_ana(
            caminho_ana
        )

        resumo = resumo_chuva(
            chuva
        )

        if chuva.empty:

            st.error(
                "Nenhuma observação diária "
                "foi encontrada."
            )

        else:

            a1, a2, a3, a4 = (
                st.columns(4)
            )

            a1.metric(
                "Período",
                (
                    f"{resumo['inicio']:%m/%Y}–"
                    f"{resumo['fim']:%m/%Y}"
                ),
            )

            a2.metric(
                "Dias disponíveis",
                f"{resumo['n_dias']:,}".replace(
                    ",",
                    ".",
                ),
            )

            a3.metric(
                "Máxima diária",
                (
                    f"{resumo['max_diaria_mm']:.1f} mm"
                ),
            )

            a4.metric(
                "Data da máxima",
                (
                    f"{resumo['data_maxima']:%d/%m/%Y}"
                ),
            )

            p1, p2, p3 = (
                st.columns(3)
            )

            p1.metric(
                "P90 chuva",
                (
                    f"{resumo['p90_mm']:.1f} mm"
                ),
            )

            p2.metric(
                "P95 chuva",
                (
                    f"{resumo['p95_mm']:.1f} mm"
                ),
            )

            p3.metric(
                "P99 chuva",
                (
                    f"{resumo['p99_mm']:.1f} mm"
                ),
            )

            st.caption(
                "Os percentis de precipitação "
                "caracterizam a série histórica e "
                "não representam limiares de inundação."
            )

            anual = (
                chuva
                .assign(
                    ano=chuva[
                        "data"
                    ].dt.year
                )
                .groupby(
                    "ano",
                    as_index=False,
                )["chuva_mm"]
                .sum()
            )

            fig_chuva = go.Figure()

            fig_chuva.add_bar(
                x=anual["ano"],
                y=anual["chuva_mm"],
                name="Precipitação anual",
            )

            fig_chuva.update_layout(
                xaxis_title="Ano",
                yaxis_title=(
                    "Precipitação acumulada (mm)"
                ),
            )

            st.plotly_chart(
                fig_chuva,
                width="stretch",
            )

            with st.expander(
                "Ver maiores chuvas diárias"
            ):

                top = chuva.nlargest(
                    10,
                    "chuva_mm",
                )[
                    [
                        "data",
                        "chuva_mm",
                    ]
                ].copy()

                top["data"] = (
                    top["data"]
                    .dt.strftime(
                        "%d/%m/%Y"
                    )
                )

                top.columns = [
                    "Data",
                    "Chuva (mm)",
                ]

                st.dataframe(
                    top,
                    width="stretch",
                    hide_index=True,
                )

    except Exception as exc:

        st.exception(exc)

    st.info(
        "A resolução diária da série histórica constitui "
        "uma limitação importante para eventos de resposta "
        "rápida. O sensor previsto para o projeto permitirá "
        "registrar a evolução do nível em intervalos muito "
        "menores e poderá apoiar futuras previsões subdiárias."
    )


# ================================================================
# 4. PROPAGAÇÃO
# ================================================================

with abas[5]:
    pagina_muskingum_demo()
    st.divider()
    st.subheader("Modelo original: simulação manual")

    st.subheader(
        "Simulação experimental de propagação"
    )

    st.caption(
        "Ponte → Poço Verde"
    )

    st.info(
        "Este módulo é complementar ao modelo preditivo D+1. "
        "Ele avalia experimentalmente a transformação "
        "nível-vazão e a propagação de um hidrograma. "
        "Os parâmetros ainda não estão calibrados."
    )

    modo_app = st.radio(
        "Modo",
        [
            "Demonstração",
            "Engenharia",
        ],
        horizontal=True,
    )

    col_a, col_b = st.columns(
        [1, 2]
    )

    with col_a:

        st.markdown(
            "### Entrada de nível"
        )

        fonte_nivel = st.radio(
            "Fonte",
            [
                "Cenário sintético",
                "Digitar série manual",
            ],
        )

        passo_min = st.selectbox(
            "Intervalo entre leituras (min)",
            [
                15,
                10,
                5,
                30,
            ],
            index=0,
        )

        if (
            fonte_nivel
            == "Cenário sintético"
        ):

            nivel_base = st.number_input(
                "Nível de base (m)",
                min_value=0.0,
                value=0.55,
                step=0.05,
            )

            nivel_pico = st.number_input(
                "Nível de pico (m)",
                min_value=0.01,
                value=1.50,
                step=0.05,
            )

            n_passos = st.slider(
                "Número de leituras",
                min_value=9,
                max_value=41,
                value=17,
                step=2,
            )

            serie_nivel = (
                gerar_niveis_sinteticos(
                    nivel_base=nivel_base,
                    nivel_pico=nivel_pico,
                    n_passos=n_passos,
                    passo_min=passo_min,
                )
            )

        else:

            texto = st.text_area(
                "Níveis em metros, um por linha",
                value=(
                    "0.55\n0.58\n0.65\n0.82\n"
                    "1.04\n1.32\n1.50\n1.42\n"
                    "1.28\n1.10\n0.94\n0.80\n0.70"
                ),
                height=220,
            )

            valores = []

            for linha in texto.splitlines():

                linha = (
                    linha.strip()
                    .replace(
                        ",",
                        ".",
                    )
                )

                if not linha:
                    continue

                try:

                    valor = float(
                        linha
                    )

                    if valor >= 0:
                        valores.append(
                            valor
                        )

                except ValueError:
                    pass

            if len(valores) < 2:

                st.error(
                    "Informe pelo menos dois "
                    "níveis válidos."
                )

                st.stop()

            serie_nivel = pd.DataFrame(
                {
                    "data_hora":
                        pd.date_range(
                            "2026-09-03 08:00",
                            periods=len(
                                valores
                            ),
                            freq=(
                                f"{passo_min}min"
                            ),
                        ),
                    "nivel_m":
                        valores,
                }
            )

        serie_nivel = (
            serie_nivel.copy()
        )

        serie_nivel[
            "delta_h_m"
        ] = serie_nivel[
            "nivel_m"
        ].diff()

        serie_nivel[
            "taxa_subida_m_h"
        ] = (
            serie_nivel[
                "delta_h_m"
            ]
            / (passo_min / 60.0)
        )

        taxa_max = (
            serie_nivel[
                "taxa_subida_m_h"
            ].max()
        )

        nivel_max = (
            serie_nivel[
                "nivel_m"
            ].max()
        )

        e1, e2 = st.columns(2)

        e1.metric(
            "Maior nível",
            f"{nivel_max:.2f} m",
        )

        e2.metric(
            "Maior taxa de elevação",
            (
                f"{taxa_max:.2f} m/h"
                if pd.notna(
                    taxa_max
                )
                else "—"
            ),
        )

        a_curva = 12.0
        h0 = 0.20
        b_exp = 1.60
        k_h = 0.50
        x = 0.20

        if modo_app == "Engenharia":

            st.markdown(
                "### Parâmetros"
            )

            a_curva = st.number_input(
                "a",
                min_value=0.001,
                value=12.0,
                step=0.5,
            )

            h0 = st.number_input(
                "H0 (m)",
                min_value=0.0,
                value=0.20,
                step=0.05,
            )

            b_exp = st.number_input(
                "b",
                min_value=0.10,
                value=1.60,
                step=0.10,
            )

            k_h = st.number_input(
                "K (h)",
                min_value=0.05,
                value=0.50,
                step=0.05,
            )

            x = st.number_input(
                "X",
                min_value=0.0,
                max_value=0.5,
                value=0.20,
                step=0.05,
            )

            st.warning(
                "Parâmetros experimentais "
                "ainda não calibrados."
            )

    with col_b:

        st.markdown(
            "### Nível de entrada"
        )

        fig_nivel = go.Figure()

        fig_nivel.add_trace(
            go.Scatter(
                x=serie_nivel[
                    "data_hora"
                ],
                y=serie_nivel[
                    "nivel_m"
                ],
                mode="lines+markers",
                name="Nível",
            )
        )

        fig_nivel.update_layout(
            xaxis_title="Horário",
            yaxis_title="Nível (m)",
            hovermode="x unified",
        )

        st.plotly_chart(
            fig_nivel,
            width="stretch",
        )

        q_ponte = nivel_para_vazao(
            serie_nivel[
                "nivel_m"
            ].to_numpy(),
            a=a_curva,
            h0=h0,
            b=b_exp,
        )

        dt_h = (
            passo_min / 60.0
        )

        try:

            q_jusante, coefs = (
                rotear_muskingum(
                    q_ponte,
                    k_h=k_h,
                    x=x,
                    dt_h=dt_h,
                )
            )

        except ValueError as exc:

            st.error(
                str(exc)
            )

            st.stop()

        resultado = (
            serie_nivel.copy()
        )

        resultado[
            "q_ponte_m3s"
        ] = q_ponte

        resultado[
            "q_poco_verde_m3s"
        ] = q_jusante

        idx_pico_ponte = int(
            np.argmax(
                q_ponte
            )
        )

        idx_pico_jus = int(
            np.argmax(
                q_jusante
            )
        )

        t_ponte = resultado.loc[
            idx_pico_ponte,
            "data_hora",
        ]

        t_jus = resultado.loc[
            idx_pico_jus,
            "data_hora",
        ]

        atraso_min = (
            (
                t_jus
                - t_ponte
            ).total_seconds()
            / 60.0
        )

        pico_ponte = float(
            np.max(
                q_ponte
            )
        )

        pico_jus = float(
            np.max(
                q_jusante
            )
        )

        atenuacao = (
            (
                1
                - pico_jus
                / pico_ponte
            )
            * 100
            if pico_ponte > 0
            else 0.0
        )

        st.markdown(
            "### Resultado experimental"
        )

        r1, r2, r3, r4 = (
            st.columns(4)
        )

        r1.metric(
            "Pico Ponte",
            f"{pico_ponte:.2f} m³/s",
        )

        r2.metric(
            "Pico Poço Verde",
            f"{pico_jus:.2f} m³/s",
        )

        r3.metric(
            "Defasagem",
            f"{atraso_min:.0f} min",
        )

        r4.metric(
            "Antecedência operacional",
            "A validar",
        )

        st.metric(
            "Atenuação simulada",
            f"{atenuacao:.1f}%",
        )

        fig_vazao = go.Figure()

        fig_vazao.add_trace(
            go.Scatter(
                x=resultado[
                    "data_hora"
                ],
                y=resultado[
                    "q_ponte_m3s"
                ],
                mode="lines+markers",
                name="Ponte",
            )
        )

        fig_vazao.add_trace(
            go.Scatter(
                x=resultado[
                    "data_hora"
                ],
                y=resultado[
                    "q_poco_verde_m3s"
                ],
                mode="lines+markers",
                name="Poço Verde",
            )
        )

        fig_vazao.update_layout(
            xaxis_title="Horário",
            yaxis_title=(
                "Vazão experimental (m³/s)"
            ),
            hovermode="x unified",
        )

        st.plotly_chart(
            fig_vazao,
            width="stretch",
        )

        st.caption(
            "A defasagem simulada entre picos "
            "não equivale automaticamente à "
            "antecedência de um alerta."
        )

        if coefs[
            "condicao_recomendada"
        ]:

            st.success(
                "Condição numérica do "
                "Muskingum atendida."
            )

        else:

            st.error(
                "Os parâmetros atuais não "
                "atendem à condição recomendada "
                "2KX ≤ Δt ≤ 2K(1-X)."
            )

        with st.expander(
            "Ver tabela calculada"
        ):

            exib = resultado.copy()

            exib[
                "data_hora"
            ] = exib[
                "data_hora"
            ].dt.strftime(
                "%d/%m/%Y %H:%M"
            )

            st.dataframe(
                exib,
                width="stretch",
                hide_index=True,
            )


# ================================================================
# 5. ENGENHARIA E VALIDAÇÃO
# ================================================================

with abas[6]:

    st.subheader(
        "Engenharia, metodologia e validação"
    )

    st.markdown(
        "### Arquitetura atual"
    )

    st.markdown(
        """
**Histórico ANA**

chuva + cota → construção e avaliação do modelo

↓

**Condição hidrológica atual**

cota atual + precipitação antecedente

↓

**Previsão meteorológica**

Open-Meteo → precipitação futura

↓

**Modelo preditivo**

condição atual + chuva antecedente +
chuva futura → **cota D+1**

↓

**Evolução futura**

sensor de nível → dados subdiários →
previsão de curto prazo e análise do pico
        """
    )

    st.markdown(
        "### Desempenho já obtido"
    )

    desempenho = pd.DataFrame(
        [
            {
                "Modelo":
                    "Persistência",
                "MAE":
                    6.03,
                "RMSE":
                    12.22,
                "R²":
                    0.722,
            },
            {
                "Modelo":
                    "Regressão sem chuva futura",
                "MAE":
                    6.72,
                "RMSE":
                    11.66,
                "R²":
                    0.747,
            },
            {
                "Modelo":
                    "Regressão + chuva futura",
                "MAE":
                    5.70,
                "RMSE":
                    10.60,
                "R²":
                    0.791,
            },
            {
                "Modelo":
                    "Random Forest + chuva futura",
                "MAE":
                    5.30,
                "RMSE":
                    11.70,
                "R²":
                    0.745,
            },
        ]
    )

    st.dataframe(
        desempenho,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Comparação realizada em conjunto de teste "
        "cronológico. O desempenho geral não implica "
        "desempenho equivalente nos eventos extremos."
    )

    st.markdown(
        "### Referências estatísticas de cota"
    )

    ref1, ref2 = st.columns(2)

    ref1.metric(
        "P90 histórico",
        "84"
    )

    ref2.metric(
        "P95 histórico",
        "≈ 97"
    )

    st.warning(
        "P90 e P95 são referências estatísticas "
        "exploratórias. Não correspondem a cotas "
        "oficiais de atenção, alerta, transbordamento "
        "ou inundação."
    )

    st.markdown(
        "### Limitações atuais"
    )

    st.markdown(
        """
- A série histórica integrada possui resolução diária.
- Alguns eventos de elevação abrupta não são adequadamente explicados pelas variáveis atualmente disponíveis.
- A chuva futura utilizada no treinamento representa chuva observada em D+1; operacionalmente, essa variável é substituída pela previsão meteorológica e incorpora erro de previsão.
- A cota atual ainda precisa ser informada manualmente.
- A precipitação antecedente recente ainda precisa ser informada manualmente.
- A curva-chave e o módulo Muskingum permanecem experimentais.
- Ainda não existe limiar hidráulico local validado para transformar a cota prevista em classe oficial de inundação.
        """
    )

    st.markdown(
        "### Papel esperado do sensor"
    )

    st.markdown(
        """
O sensor de nível permitirá substituir a entrada manual
por observações frequentes do comportamento do Rio Soberbo.

Com uma série subdiária, poderão ser incorporadas ao modelo
informações como:

**cota atual → variação recente → velocidade de subida →
aceleração da subida → precipitação prevista → cota futura**

Essa evolução poderá permitir a investigação de previsões
em horizontes de horas e da janela provável de ocorrência
do pico, desde que exista volume suficiente de dados para
calibração e validação.
        """
    )

    st.markdown(
        "### Curva-chave e propagação"
    )

    st.markdown(
        r"""
Curva-chave experimental:

\[
Q = a(H-H_0)^b
\]

Muskingum:

\[
S = K[XI + (1-X)O]
\]

e:

\[
O_{t+\Delta t}
=
C_0 I_{t+\Delta t}
+
C_1 I_t
+
C_2 O_t
\]

Esses componentes permanecem módulos de engenharia
complementares e dependem de calibração com observações
adequadas antes de qualquer interpretação operacional.
        """
    )

    st.divider()

    st.warning(
        "O SoberboPrev é uma prova de conceito acadêmica. "
        "As previsões e referências apresentadas não devem "
        "ser utilizadas para emissão de alertas oficiais."
    )