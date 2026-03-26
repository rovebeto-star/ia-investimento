import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARQ_FIXOS = os.path.join(BASE_DIR, "fixos.csv")
ARQ_EUA = os.path.join(BASE_DIR, "scanner_eua.csv")
ARQ_BR = os.path.join(BASE_DIR, "scanner_br.csv")

st.set_page_config(
    page_title="Painel de Investimentos",
    page_icon="📈",
    layout="wide"
)


def carregar_csv(caminho):
    if not os.path.exists(caminho):
        return pd.DataFrame()

    try:
        df = pd.read_csv(caminho)
        df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Erro ao ler {os.path.basename(caminho)}: {e}")
        return pd.DataFrame()


def preparar_df(df):
    if df.empty:
        return df

    df = df.copy()

    colunas_numericas = [
        "Preco", "Variacao", "Score",
        "Prob_Alta", "Prob_Baixa", "Confianca_IA",
        "Acerto", "Operacoes", "Retorno"
    ]

    for col in colunas_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(how="all")
    return df


def classificar_forca(df):
    if df.empty:
        return df

    df = df.copy()

    def forca(row):
        sinal = row.get("Sinal", "")
        score = row.get("Score", 0)
        confianca = row.get("Confianca_IA", 0)

        if sinal == "COMPRA" and (score >= 60 or confianca >= 65):
            return "🔥 FORTE"
        elif sinal == "COMPRA":
            return "✅ COMPRA"
        elif sinal == "VENDA":
            return "🔴 VENDA"
        elif sinal == "AGUARDAR":
            return ""
        return ""

    df["Forca"] = df.apply(forca, axis=1)
    return df


def estilo_sinal(valor):
    if valor == "COMPRA":
        return "background-color: #0f5132; color: #00ff88; font-weight: bold;"
    elif valor == "VENDA":
        return "background-color: #5a0f0f; color: #ff6b6b; font-weight: bold;"
    elif valor == "AGUARDAR":
        return "background-color: #5a4a0f; color: #ffd700; font-weight: bold;"
    return ""


def aplicar_estilo(df):
    if df.empty:
        return df

    styled = df.style

    if "Sinal" in df.columns:
        styled = styled.map(estilo_sinal, subset=["Sinal"])

    return styled


def contar_sinais(*dfs):
    compra = venda = aguardar = 0

    for df in dfs:
        if not df.empty and "Sinal" in df.columns:
            compra += int((df["Sinal"] == "COMPRA").sum())
            venda += int((df["Sinal"] == "VENDA").sum())
            aguardar += int((df["Sinal"] == "AGUARDAR").sum())

    return compra, venda, aguardar


def top_geral(scanner_eua, scanner_br, top_n=10):
    frames = []

    if not scanner_eua.empty:
        eua = scanner_eua.copy()
        eua["Mercado"] = "EUA"
        frames.append(eua)

    if not scanner_br.empty:
        br = scanner_br.copy()
        br["Mercado"] = "Brasil"
        frames.append(br)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    colunas_ordem = [c for c in ["Score", "Confianca_IA", "Retorno", "Acerto"] if c in df.columns]
    if colunas_ordem:
        df = df.sort_values(by=colunas_ordem, ascending=[False] * len(colunas_ordem))

    df.insert(0, "Rank", range(1, len(df) + 1))
    return df.head(top_n)


def colunas_existentes(df, lista):
    return [c for c in lista if c in df.columns]


fixos = classificar_forca(preparar_df(carregar_csv(ARQ_FIXOS)))
scanner_eua = classificar_forca(preparar_df(carregar_csv(ARQ_EUA)))
scanner_br = classificar_forca(preparar_df(carregar_csv(ARQ_BR)))
ranking = top_geral(scanner_eua, scanner_br, top_n=10)

st.title("📊 Painel de Investimentos")
st.caption(f"Última visualização: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

compra, venda, aguardar = contar_sinais(fixos, scanner_eua, scanner_br)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🟢 Compra", compra)

with col2:
    st.metric("🔴 Venda", venda)

with col3:
    st.metric("🟡 Aguardar", aguardar)

with col4:
    st.metric("📌 Total analisado", compra + venda + aguardar)

st.divider()

st.subheader("🚀 Oportunidades Fortes")

if not ranking.empty:
    filtros_fortes = (ranking["Sinal"] == "COMPRA")
    if "Score" in ranking.columns:
        filtros_fortes = filtros_fortes & (ranking["Score"] >= 60)

    fortes = ranking[filtros_fortes].copy()

    if not fortes.empty:
        colunas_fortes = colunas_existentes(
            fortes,
            ["Rank", "Mercado", "Ativo", "Forca", "Sinal", "Preco", "Variacao", "Score",
             "Confianca_IA", "Acerto", "Retorno"]
        )
        st.dataframe(
            aplicar_estilo(fortes[colunas_fortes]),
            width="stretch",
            height=220,
            hide_index=True
        )
    else:
        st.warning("Nenhuma oportunidade forte no momento.")
else:
    st.warning("Ranking ainda não disponível.")

st.divider()

col_esq, col_dir = st.columns([2, 1])

with col_esq:
    st.subheader("🏆 Top 10 Geral")
    if not ranking.empty:
        exibir = ranking.copy()
        colunas = colunas_existentes(
            exibir,
            ["Rank", "Mercado", "Ativo", "Forca", "Sinal", "Preco", "Variacao", "Score",
             "Confianca_IA", "Acerto", "Operacoes", "Retorno"]
        )
        st.dataframe(
            aplicar_estilo(exibir[colunas]),
            width="stretch",
            height=380,
            hide_index=True
        )
    else:
        st.info("Nenhum ranking disponível ainda.")

with col_dir:
    st.subheader("📈 Distribuição de Sinais")

    graf_df = pd.DataFrame({
        "Sinal": ["COMPRA", "VENDA", "AGUARDAR"],
        "Quantidade": [compra, venda, aguardar]
    })

    fig = px.pie(
        graf_df,
        names="Sinal",
        values="Quantidade",
        hole=0.45
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")

st.divider()

st.subheader("🎯 Filtros")

f1, f2 = st.columns(2)

with f1:
    filtro_sinal = st.multiselect(
        "Filtrar por sinal",
        options=["COMPRA", "VENDA", "AGUARDAR"],
        default=["COMPRA", "VENDA", "AGUARDAR"]
    )

with f2:
    mercado_escolhido = st.selectbox(
        "Mercado do ranking",
        options=["Todos", "EUA", "Brasil"]
    )

st.divider()

st.subheader("📌 Ativos Fixos")
if not fixos.empty:
    df_fixos = fixos.copy()
    if "Sinal" in df_fixos.columns:
        df_fixos = df_fixos[df_fixos["Sinal"].isin(filtro_sinal)]

    colunas_fixos = colunas_existentes(
        df_fixos,
        ["Ativo", "Forca", "Sinal", "Preco", "Variacao", "Prob_Alta", "Prob_Baixa",
         "Confianca_IA", "Acerto", "Operacoes", "Retorno"]
    )
    st.dataframe(
        aplicar_estilo(df_fixos[colunas_fixos]),
        width="stretch",
        height=320,
        hide_index=True
    )
else:
    st.info("Arquivo fixos.csv ainda não encontrado.")

st.divider()

st.subheader("🇺🇸 Scanner EUA")
if not scanner_eua.empty:
    df_eua = scanner_eua.copy()
    if "Sinal" in df_eua.columns:
        df_eua = df_eua[df_eua["Sinal"].isin(filtro_sinal)]

    colunas_eua = colunas_existentes(
        df_eua,
        ["Ativo", "Forca", "Sinal", "Preco", "Variacao", "Score", "Prob_Alta", "Prob_Baixa",
         "Confianca_IA", "Acerto", "Operacoes", "Retorno"]
    )
    st.dataframe(
        aplicar_estilo(df_eua[colunas_eua]),
        width="stretch",
        height=320,
        hide_index=True
    )
else:
    st.info("Arquivo scanner_eua.csv ainda não encontrado.")

st.divider()

st.subheader("🇧🇷 Scanner Brasil")
if not scanner_br.empty:
    df_br = scanner_br.copy()
    if "Sinal" in df_br.columns:
        df_br = df_br[df_br["Sinal"].isin(filtro_sinal)]

    colunas_br = colunas_existentes(
        df_br,
        ["Ativo", "Forca", "Sinal", "Preco", "Variacao", "Score", "Prob_Alta", "Prob_Baixa",
         "Confianca_IA", "Acerto", "Operacoes", "Retorno"]
    )
    st.dataframe(
        aplicar_estilo(df_br[colunas_br]),
        width="stretch",
        height=320,
        hide_index=True
    )
else:
    st.info("Arquivo scanner_br.csv ainda não encontrado.")

st.divider()

st.subheader("📊 Ranking Filtrado")
if not ranking.empty:
    ranking_filtrado = ranking.copy()

    if "Sinal" in ranking_filtrado.columns:
        ranking_filtrado = ranking_filtrado[ranking_filtrado["Sinal"].isin(filtro_sinal)]

    if mercado_escolhido != "Todos":
        ranking_filtrado = ranking_filtrado[ranking_filtrado["Mercado"] == mercado_escolhido]

    colunas_rank = colunas_existentes(
        ranking_filtrado,
        ["Rank", "Mercado", "Ativo", "Forca", "Sinal", "Preco", "Variacao", "Score",
         "Prob_Alta", "Prob_Baixa", "Confianca_IA", "Acerto", "Operacoes", "Retorno"]
    )
    st.dataframe(
        aplicar_estilo(ranking_filtrado[colunas_rank]),
        width="stretch",
        height=350,
        hide_index=True
    )
else:
    st.info("Ranking ainda não disponível.")