import os
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

CHANNEL_ID = 2406014

@st.cache_data(ttl=60, show_spinner=False)
def consultar_sensor():
    try:
        chave = st.secrets.get('THINGSPEAK_READ_API_KEY', '')
    except Exception:
        chave = ''
    chave = chave or os.environ.get('THINGSPEAK_READ_API_KEY', '')
    parametros = {'results': 300}
    if chave:
        parametros['api_key'] = chave
    resposta = requests.get(f'https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json', params=parametros, timeout=15)
    resposta.raise_for_status()
    dados = resposta.json()
    return dados.get('channel', {}), pd.DataFrame(dados.get('feeds', []))

def painel_sensor():
    st.subheader('Monitoramento do Rio Soberbo')
    st.caption('Defesa Civil • Dados do sensor ThingSpeak • Protótipo acadêmico, sem alertas oficiais')
    if st.button('Atualizar leituras', key='atualizar_sensor'):
        consultar_sensor.clear()
        st.rerun()
    try:
        canal, bruto = consultar_sensor()
    except Exception as erro:
        st.error('Sem comunicação com o sensor. Não é possível avaliar a condição atual do rio.')
        with st.expander('Diagnóstico da conexão'):
            st.code(str(erro))
        return
    campos = [f'field{i}' for i in range(1, 9) if f'field{i}' in bruto.columns and bruto[f'field{i}'].notna().any()]
    if not campos:
        st.warning('A API respondeu, mas não há leituras disponíveis nos campos do canal.')
        st.json({k: canal.get(k) for k in ['id','name','field1','field2','field3','last_entry_id']})
        return
    with st.expander('Configuração técnica do sensor (pendente de validação)', expanded=False):
        campo = st.selectbox('Campo com medições', campos, format_func=lambda c: f'{c} — {canal.get(c) or "sem descrição"}')
        tipo = st.selectbox('O que este campo mede?', ['Não identificado', 'Nível de água', 'Distância sensor-superfície'])
        unidade = st.selectbox('Unidade', ['Não confirmada', 'cm', 'm', 'mm'])
        referencia = st.number_input('Altura do sensor acima do zero local (mesma unidade)', value=0.0, help='Somente para distância; precisa ser medida e validada.') if tipo == 'Distância sensor-superfície' else None
        st.caption('Não insira limiares fictícios como se fossem alertas reais. Faixas críticas dependem de validação local.')
    tabela = bruto[['created_at', campo]].copy()
    tabela['instante'] = pd.to_datetime(tabela['created_at'], utc=True, errors='coerce').dt.tz_convert('America/Sao_Paulo')
    tabela['bruto'] = pd.to_numeric(tabela[campo], errors='coerce')
    tabela = tabela.dropna(subset=['instante', 'bruto']).sort_values('instante').drop_duplicates('instante')
    if tabela.empty:
        st.warning('O campo selecionado não possui leituras numéricas válidas.')
        return
    validado = tipo != 'Não identificado' and unidade != 'Não confirmada' and (tipo != 'Distância sensor-superfície' or referencia is not None and referencia > 0)
    tabela['nivel'] = referencia - tabela['bruto'] if tipo == 'Distância sensor-superfície' and validado else tabela['bruto']
    ultima = tabela.iloc[-1]
    atraso = (pd.Timestamp.now(tz='America/Sao_Paulo') - ultima['instante']).total_seconds()/60
    st.caption(f'Canal {CHANNEL_ID} • Última leitura: {ultima["instante"]:%d/%m/%Y %H:%M:%S} (Brasília)')
    if atraso > 60:
        st.error(f'LEITURA DESATUALIZADA — há {atraso:.0f} minutos. Não interpretar como condição atual.')
    elif atraso < -5:
        st.warning('Horário da leitura está no futuro: verificar relógio e fuso do dispositivo.')
    if not validado:
        st.warning('Variável, unidade ou referência não confirmada. Valores brutos não representam necessariamente o nível do rio.')
    else:
        st.info('Nível calculado com a configuração informada, ainda não validado para uso operacional.')
    col1, col2, col3 = st.columns(3)
    col1.metric('Nível calculado' if validado else 'Última leitura bruta', f'{ultima["nivel"]:.2f} {unidade}' if validado else f'{ultima["bruto"]:.2f} (unidade desconhecida)')
    if len(tabela) >= 2:
        anterior = tabela.iloc[-2]
        dt = (ultima['instante']-anterior['instante']).total_seconds()/3600
        if dt > 0 and validado:
            taxa = (ultima['nivel']-anterior['nivel'])/dt
            col2.metric('Variação por hora', f'{taxa:+.2f} {unidade}/h')
            col3.metric('Tendência entre leituras', 'Subindo' if taxa > 0 else ('Descendo' if taxa < 0 else 'Estável'))
        else:
            col2.metric('Variação', 'A validar')
            col3.metric('Tendência do rio', 'Indisponível')
    else:
        col2.metric('Variação', 'Sem histórico')
        col3.metric('Tendência', 'Indisponível')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tabela['instante'], y=tabela['nivel'] if validado else tabela['bruto'], mode='lines+markers', name='Leituras'))
    fig.update_layout(title='Histórico recente do sensor', xaxis_title='Data e hora (Brasília)', yaxis_title=f'Nível ({unidade})' if validado else 'Valor bruto (unidade não identificada)', height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.info('Faixas de atenção, alerta e inundação: não configuradas. Necessária validação hidráulica e referência altimétrica do sensor.')
    st.download_button('Exportar leituras CSV', tabela[['instante','bruto']].to_csv(index=False).encode('utf-8'), file_name='soberbo_leituras_sensor.csv', mime='text/csv')
    with st.expander('Diagnóstico dos campos da API'):
        st.json({k: canal.get(k) for k in ['id','name','description','field1','field2','field3','field4','field5','field6','field7','field8','last_entry_id']})
        st.dataframe(bruto.tail(10), use_container_width=True)
