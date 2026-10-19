# SoberboPrev V1

Protótipo Streamlit para análise experimental de propagação de eventos hidrológicos no Rio Soberbo.

## O que já está implementado

- leitura e transformação do CSV de chuva da ANA (estação 2242094);
- mapa simples com Ponte e Poço Verde;
- entrada sintética ou manual de níveis em intervalos regulares;
- curva-chave experimental `Q = a(H-H0)^b`;
- roteamento pelo método de Muskingum;
- cálculo de pico, defasagem entre picos e atenuação;
- página com equações e lista de itens que precisam ser validados.

## Pontos usados

- Ponte/sensor: `-22.4646320, -43.0428580`
- Poço Verde: `-22.49135549500824, -43.004712129626256`

A distância em linha reta é de aproximadamente 4,92 km. Ela NÃO representa o comprimento do rio.

## Como executar

Abra o terminal dentro da pasta do projeto e rode:

    pip install -r requirements.txt
    streamlit run app.py

## Status técnico

A precipitação histórica é observada.  
A série de nível e os parâmetros hidráulicos são experimentais.  
A V1 não está calibrada para uso operacional e não deve ser utilizada como alerta oficial.
