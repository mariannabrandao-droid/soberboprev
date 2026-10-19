from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.sensor_thingspeak import consultar_sensor
from src.rating_curve import nivel_para_vazao
from src.muskingum import rotear_muskingum

TZ='America/Sao_Paulo'

@st.cache_data(ttl=180,show_spinner=False)
def dados_demo():
    try:
        _, bruto=consultar_sensor()
        df=bruto[['created_at','field1']].copy()
        df['original']=pd.to_datetime(df['created_at'],utc=True,errors='coerce').dt.tz_convert(TZ)
        df['valor']=pd.to_numeric(df['field1'],errors='coerce')
        df=df.dropna(subset=['original','valor']).sort_values('original').drop_duplicates('original')
        if len(df)<3: raise ValueError('Registros insuficientes')
        origem='ThingSpeak — sensor externo usado apenas como série ilustrativa'
    except Exception:
        t=np.arange(96)
        df=pd.DataFrame({'original':pd.date_range('2026-01-01',periods=96,freq='15min',tz=TZ),'valor':110+17*np.sin(t/10)+28*np.exp(-((t-68)/12)**2)})
        origem='Série sintética local — API indisponível'
    agora=pd.Timestamp.now(tz=TZ).floor('min')
    df['instante']=agora-(df['original'].max()-df['original'])
    return df.reset_index(drop=True),origem

def parametros():
    # Chamada apenas uma vez por execução. Evita StreamlitDuplicateElementKey.
    with st.sidebar.expander('Configurar cenário demonstrativo',expanded=False):
        st.caption('Hipóteses ilustrativas; não são níveis oficiais do Rio Soberbo.')
        escala=st.number_input('Unidades brutas por metro',min_value=0.01,value=100.0,step=10.0,key='v4_escala')
        base=st.number_input('Deslocamento vertical (m)',value=0.0,step=0.1,key='v4_base')
        at=st.number_input('Atenção (m)',min_value=0.0,value=1.30,step=0.05,key='v4_at')
        retirada=st.number_input('Retirada preventiva (m)',min_value=0.0,value=1.45,step=0.05,key='v4_ret')
        crit=st.number_input('Crítico (m)',min_value=0.0,value=1.65,step=0.05,key='v4_crit')
    return escala,base,at,retirada,crit

def preparar():
    if 'v4_parametros' not in st.session_state:
        raise RuntimeError('Inicialize os parâmetros no app.py antes de abrir as abas')
    df,origem=dados_demo()
    escala,base,at,ret,crit=st.session_state['v4_parametros']
    df=df.copy()
    df['nivel_m']=base+df['valor']/escala
    return df,origem,at,ret,crit

def classificar(h,at,ret,crit):
    if not 0<=at<ret<crit: return 'CONFIGURAÇÃO INVÁLIDA','#64748b','Ajustar os limiares do cenário.'
    if h>=crit: return 'CRÍTICO','#bd3f45','Simulação: condição crítica atingida.'
    if h>=ret: return 'RETIRADA PREVENTIVA','#dd792d','Simulação: iniciar protocolo de retirada de banhistas.'
    if h>=at: return 'ATENÇÃO','#c99b25','Simulação: preparar equipe e intensificar observação.'
    return 'NORMAL','#21845b','Simulação: nível abaixo da faixa de atenção.'

def grafico(df,at,ret,crit):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df['instante'],y=df['nivel_m'],name='Nível ilustrativo',mode='lines',fill='tozeroy',fillcolor='rgba(32,107,180,.12)',line=dict(color='#206bb4',width=2.5)))
    if 0<=at<ret<crit:
        for y,cor,nome in [(at,'#c99b25','Atenção'),(ret,'#dd792d','Retirada preventiva'),(crit,'#bd3f45','Crítico')]:
            fig.add_hline(y=y,line_dash='dash',line_color=cor,annotation_text=nome,annotation_position='top left')
    fig.update_layout(template='plotly_white',height=365,margin=dict(l=15,r=15,t=20,b=20),xaxis_title='Horário ilustrativo (Brasília)',yaxis_title='Nível hipotético (m)',hovermode='x unified',legend=dict(orientation='h',y=1.12),uirevision='soberbo-v4')
    fig.update_xaxes(rangeslider_visible=True)
    return fig

def pagina_resumo_demo():
    # Resumo ilustrativo, não operacional.
    df, origem, at, ret, crit = preparar()
    if len(df) < 2:
        st.warning('Sem dados suficientes para demonstrar o painel.')
        return
    df = df.sort_values('instante').drop_duplicates('instante').copy()
    ultimo = df.iloc[-1]
    h = float(ultimo['nivel_m'])
    nome, cor, orientacao = classificar(h, at, ret, crit)
    instante = ultimo['instante']
    st.markdown('### Rio Soberbo | Situação do rio')
    st.caption('Ponte → Poço Verde · Guapimirim (RJ) · Painel de demonstração')
    st.info('**CENÁRIO FICTÍCIO:** valores de outro sensor ou gerados artificialmente, com datas deslocadas. Não representa o nível real do Rio Soberbo e não serve para orientar a retirada de pessoas.')

    acoes = {
        'NORMAL':'Nível abaixo da faixa de atenção neste cenário.',
        'ATENÇÃO':'No exercício, intensificar a observação e preparar a equipe.',
        'RETIRADA PREVENTIVA':'No exercício, iniciar a retirada preventiva dos banhistas.',
        'CRÍTICO':'No exercício, manter as pessoas afastadas das áreas de risco.',
        'CONFIGURAÇÃO INVÁLIDA':'Revisar os limites ilustrativos na barra lateral.'
    }
    st.markdown(f'''<div style="background:{cor};color:white;border-radius:10px;padding:18px 22px;margin:8px 0 18px">
    <div style="font-size:12px;font-weight:700;letter-spacing:.06em">CLASSIFICAÇÃO DO CENÁRIO · SIMULADA</div>
    <div style="font-size:28px;font-weight:800;margin:5px 0">{nome}</div>
    <div style="font-size:15px">{acoes[nome]}</div></div>''',unsafe_allow_html=True)

    # Apenas medições existentes no período. Nunca completar uma janela
    # insuficiente com observações externas à janela solicitada.
    janelas = {'1 hora': 1, '3 horas': 3, '6 horas': 6,
               '12 horas': 12, '24 horas': 24, '48 horas': 48,
               'Todo o histórico': None}
    st.markdown('#### Como o nível está mudando?')
    janela = st.radio('Período do gráfico (simulado)', list(janelas),
                      index=4, horizontal=True, key='v7_janela')
    horas = janelas[janela]
    dfg = df.copy() if horas is None else df.loc[
        (df['instante'] >= instante - pd.Timedelta(hours=horas)) &
        (df['instante'] <= instante)
    ].copy()
    valido = len(dfg) >= 2
    diferenca = float(dfg['nivel_m'].iloc[-1] - dfg['nivel_m'].iloc[0]) if valido else np.nan
    duracao_h = ((dfg['instante'].iloc[-1] - dfg['instante'].iloc[0]).total_seconds() / 3600) if valido else np.nan
    taxa_media = diferenca / duracao_h if valido and duracao_h > 0 else np.nan
    tendencia = ('Dados insuficientes' if not valido else
                 'Praticamente estável →' if abs(diferenca) < 0.01 else
                 'Subindo ↑' if diferenca > 0 else 'Descendo ↓')
    cols = st.columns(3)
    cols[0].metric('Nível ilustrativo mais recente', f'{h:.2f} m')
    cols[1].metric('Variação nas leituras disponíveis',
                   f'{diferenca:+.2f} m' if valido else 'Indisponível',
                   help='Diferença entre a primeira e a última leitura efetivamente presentes no período.')
    cols[2].metric('Tendência das leituras', tendencia)
    if not valido:
        st.warning('Há menos de duas leituras na janela escolhida. Não é possível calcular a tendência. Selecione um período maior.')
    elif horas is not None:
        st.caption(f'{len(dfg)} leituras no período selecionado; intervalo efetivamente coberto: {duracao_h:.1f} h. A variação não representa necessariamente toda a janela.')
    if nome != 'CONFIGURAÇÃO INVÁLIDA':
        proximo = (f'{at-h:.2f} m até a atenção' if h < at else
                   f'{ret-h:.2f} m até a retirada' if h < ret else
                   f'{crit-h:.2f} m até o crítico' if h < crit else
                   f'{h-crit:.2f} m acima do crítico')
        st.caption(f'Distância até a faixa: {proximo}. ' +
                   (f'Variação média no intervalo observado: {taxa_media:+.2f} m/h.'
                    if np.isfinite(taxa_media) else 'Variação média indisponível.'))

    fig = go.Figure()
    if len(dfg):
        plot_df = dfg.sort_values('instante').copy()
        # Interrompe a linha em lacunas maiores que 3 vezes a cadência
        # mediana positiva, em vez de representar medições inexistentes.
        gaps = plot_df['instante'].diff().dt.total_seconds()
        positivos = gaps[gaps > 0]
        if not positivos.empty:
            limiar = max(float(positivos.median()) * 3, 60.0)
            quebrado = plot_df['nivel_m'].astype(float).copy()
            # Segmentos separados; pontos preservados no hover.
            segmentos = (gaps > limiar).fillna(False).cumsum()
            for _, grupo in plot_df.groupby(segmentos):
                fig.add_trace(go.Scatter(
                    x=grupo['instante'], y=grupo['nivel_m'],
                    mode='lines+markers', line=dict(color='#1264a3', width=2.5),
                    marker=dict(size=5), showlegend=False,
                    hovertemplate='%{x|%d/%m %H:%M}<br>Nível fictício: %{y:.2f} m<extra></extra>'))
        else:
            fig.add_trace(go.Scatter(x=plot_df['instante'], y=plot_df['nivel_m'],
                                     mode='markers', marker=dict(color='#1264a3', size=7),
                                     showlegend=False))
    if 0 <= at < ret < crit:
        for valor, cor_linha, rotulo in [(at, '#bd9020', 'Atenção'),
                                         (ret, '#df7b2e', 'Retirada preventiva'),
                                         (crit, '#bd3f45', 'Crítico')]:
            fig.add_hline(y=valor, line_dash='dash', line_color=cor_linha,
                          annotation_text=rotulo, annotation_position='top left')
    limites = [at, ret, crit] if 0 <= at < ret < crit else []
    niveis = dfg['nivel_m'].dropna().tolist() + limites
    if niveis:
        lo, hi = min(niveis), max(niveis)
        margem = max((hi-lo)*0.15, 0.08)
        fig.update_yaxes(range=[max(0, lo-margem), hi+margem])
    if horas is not None:
        fig.update_xaxes(range=[instante-pd.Timedelta(hours=horas), instante])
    fig.update_layout(template='plotly_white', height=370,
                      margin=dict(l=8, r=12, t=25, b=5),
                      showlegend=False, hovermode='x unified',
                      xaxis_title='Horário ilustrativo',
                      yaxis_title='Nível fictício (m)', font=dict(size=12))
    st.plotly_chart(fig, use_container_width=True, key='v7_grafico')
    st.caption('Pontos azuis: leituras disponíveis no cenário. Linhas interrompidas indicam lacunas. Faixas coloridas: limites hipotéticos. Não há previsão futura neste gráfico.')

    st.markdown('#### Quando seria necessário agir?')
    st.markdown('**Normal:** acompanhar.  **Atenção:** preparar a equipe.  **Retirada preventiva:** retirar banhistas.  **Crítico:** restringir o acesso às áreas de risco. **Todas as faixas são fictícias.**')
    with st.expander('Ver mapa dos pontos do estudo'):
        st.map(pd.DataFrame([{'lat':-22.464632,'lon':-43.042858},{'lat':-22.491355495,'lon':-43.00471213}]),zoom=11)
        st.caption('Ponte e Poço Verde são pontos de referência, não áreas de inundação.')
    with st.expander('Ver origem dos dados e limitações'):
        st.write(f'Origem: {origem}. Último horário ilustrativo: {instante:%d/%m/%Y %H:%M}.')
        st.write('Os dados externos são transformados por escala arbitrária. A previsão D+1 e a propagação por Muskingum são módulos experimentais separados; nenhuma das duas está validada para avisos de cheia.')

def pagina_sensor_demo():
    df,origem,at,ret,crit=preparar()
    st.subheader('Sensor | Dados e transformações')
    st.info(origem)
    st.write('A série é usada como cenário fictício. Os horários são deslocados para a data da simulação, preservando intervalos, e os valores convertidos por escala hipotética.')
    st.plotly_chart(grafico(df,at,ret,crit),use_container_width=True,key='v4_plot_sensor')
    st.dataframe(df[['original','instante','valor','nivel_m']].tail(40),use_container_width=True,hide_index=True)
    st.download_button('Exportar série ilustrativa',df[['instante','valor','nivel_m']].to_csv(index=False).encode(),file_name='soberbo_demonstracao.csv',mime='text/csv',key='v4_export')

def pagina_muskingum_demo():
    st.subheader('Muskingum | Simulação de propagação Ponte → Poço Verde')
    st.warning('Curva-chave e parâmetros hidráulicos hipotéticos. Não há previsão operacional de chegada da cheia.')
    df,origem,_,_,_=preparar()
    a,b,c=st.columns(3)
    k=a.number_input('K (h)',value=0.50,min_value=0.01,step=0.05,key='v4_k')
    x=b.number_input('X',value=0.20,min_value=0.0,max_value=0.5,step=0.05,key='v4_x')
    passo=c.selectbox('Passo (min)',[5,10,15,30,60],index=2,key='v4_dt')
    serie=df.set_index('instante')['nivel_m'].resample(f'{passo}min').mean().interpolate('time').dropna()
    if len(serie)<3 or len(serie)>3000: st.warning('Série insuficiente ou excessivamente longa para a simulação.');return
    q=nivel_para_vazao(serie.to_numpy(),a=12.0,h0=0.20,b=1.60)
    try: jus,coef=rotear_muskingum(q,k_h=k,x=x,dt_h=passo/60)
    except Exception as exc: st.error(f'Falha no roteamento: {exc}');return
    if not coef.get('condicao_recomendada',False): st.warning('Condição recomendada 2KX ≤ Δt ≤ 2K(1-X) não atendida.')
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=serie.index,y=q,name='Ponte (simulado)',line=dict(color='#206bb4')))
    fig.add_trace(go.Scatter(x=serie.index,y=jus,name='Poço Verde (simulado)',line=dict(color='#dd792d')))
    fig.update_layout(template='plotly_white',height=370,xaxis_title='Horário ilustrativo',yaxis_title='Vazão simulada (m³/s)',hovermode='x unified')
    st.plotly_chart(fig,use_container_width=True,key='v4_plot_muskingum')
    st.caption('A transformação nível–vazão usa parâmetros ilustrativos. O atraso calculado não deve ser usado como tempo de resposta real.')
