import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText

import pandas as pd
import ta
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARQ_HISTORICO = os.path.join(BASE_DIR, "historico_modelo.csv")
ARQ_FIXOS = os.path.join(BASE_DIR, "fixos.csv")
ARQ_EUA = os.path.join(BASE_DIR, "scanner_eua.csv")
ARQ_BR = os.path.join(BASE_DIR, "scanner_br.csv")

# =========================
# CONFIG EMAIL
# =========================
EMAIL_REMETENTE = os.getenv("EMAIL_REMETENTE", "")
EMAIL_SENHA_APP = os.getenv("EMAIL_SENHA_APP", "")
EMAIL_DESTINO = os.getenv("EMAIL_DESTINO", "")

# =========================
# ATIVOS FIXOS
# =========================
ativos = [
    # EUA
    "DAR", "NVDA", "NFLX", "PAGS", "MU",
    "SLV", "QQQ", "VTI", "MSFT", "ARGT",
    # Brasil
    "AUAU3.SA", "AZEV4.SA", "NVDC34.SA", "PRIO3.SA", "PETR4.SA", "PETR3.SA", "BPAC11.SA",
]

# =========================
# UNIVERSO DO SCANNER
# =========================
ativos_scan_eua = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA",
    "GOOGL", "AMD", "INTC", "NFLX", "MU", "PAGS",
    "DAR", "JPM", "BAC", "XOM", "CVX", "KO", "PEP",
    "WMT", "COST", "BA", "DIS", "PYPL", "UBER",
    "QQQ", "SPY", "VTI", "DIA", "IWM", "SLV",
]

ativos_scan_br = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA",
    "BBAS3.SA", "ABEV3.SA", "WEGE3.SA", "RENT3.SA",
    "B3SA3.SA", "ELET6.SA", "SUZB3.SA", "RAIL3.SA",
    "GGBR4.SA", "USIM5.SA", "CSNA3.SA", "MGLU3.SA",
    "LREN3.SA", "PRIO3.SA", "VIVT3.SA", "RADL3.SA",
    "AUAU3.SA", "AZEV4.SA", "NVDC34.SA",
]


# =========================
# EMAIL
# =========================
def enviar_email(msg: str) -> None:
    if not (EMAIL_REMETENTE and EMAIL_SENHA_APP and EMAIL_DESTINO):
        print("Credenciais de e-mail não configuradas. E-mail não enviado.")
        return

    mensagem = MIMEText(msg, "plain", "utf-8")
    mensagem["Subject"] = "Sinais de Investimento"
    mensagem["From"] = EMAIL_REMETENTE
    mensagem["To"] = EMAIL_DESTINO

    with smtplib.SMTP("smtp.gmail.com", 587) as servidor:
        servidor.starttls()
        servidor.login(EMAIL_REMETENTE, EMAIL_SENHA_APP)
        servidor.send_message(mensagem)


# =========================
# BAIXAR DADOS
# =========================
def baixar_dados(ativo: str, period: str = "6mo", interval: str = "1d"):
    df = yf.download(
        ativo,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()
    df["Close"] = pd.Series(df["Close"]).astype(float).squeeze()
    return df


# =========================
# INDICADORES
# =========================
def preparar_dados(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df = df.copy()
    df["mm9"] = close.rolling(9).mean()
    df["mm21"] = close.rolling(21).mean()
    df["mm50"] = close.rolling(50).mean()
    df["rsi"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    macd = ta.trend.MACD(close=close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    df["atr"] = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()

    df["vol_media_20"] = volume.rolling(20).mean()
    df["target"] = (close.shift(-2) > close).astype(int)

    return df.dropna().copy()


# =========================
# SINAL
# =========================
def gerar_sinal(df: pd.DataFrame) -> dict:
    if len(df) < 60:
        return {
            "sinal": "DADOS INSUFICIENTES",
            "prob_alta": 0.0,
            "prob_baixa": 0.0,
            "confianca": 0.0,
        }

    features = ["mm9", "mm21", "mm50", "rsi", "macd", "macd_signal", "atr", "vol_media_20"]
    X = df[features]
    y = df["target"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    last = X.tail(1)
    prob = model.predict_proba(last)[0]

    prob_baixa = round(float(prob[0]) * 100, 2)
    prob_alta = round(float(prob[1]) * 100, 2)
    confianca = round(max(prob_alta, prob_baixa), 2)

    close = df["Close"].iloc[-1]
    mm9 = df["mm9"].iloc[-1]
    mm21 = df["mm21"].iloc[-1]
    mm50 = df["mm50"].iloc[-1]
    rsi = df["rsi"].iloc[-1]
    macd = df["macd"].iloc[-1]
    macd_signal = df["macd_signal"].iloc[-1]

    compra_forte = (
        prob_alta >= 60
        and close > mm21
        and mm9 > mm21 > mm50
        and 50 <= rsi <= 65
        and macd > macd_signal
    )

    venda_forte = (
        prob_baixa >= 60
        and close < mm21
        and mm9 < mm21 < mm50
        and 35 <= rsi <= 50
        and macd < macd_signal
    )

    if compra_forte:
        sinal = "COMPRA"
    elif venda_forte:
        sinal = "VENDA"
    else:
        sinal = "AGUARDAR"

    return {
        "sinal": sinal,
        "prob_alta": prob_alta,
        "prob_baixa": prob_baixa,
        "confianca": confianca,
    }


# =========================
# BACKTEST
# =========================
def backtest(df: pd.DataFrame):
    df = df.copy()

    if len(df) < 60:
        return None

    resultados = []

    for i in range(40, len(df) - 2):
        treino = df.iloc[:i].copy()
        teste = df.iloc[i:i + 1].copy()

        X_train = treino[["mm9", "mm21", "rsi"]]
        y_train = treino["target"]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        pred = model.predict(teste[["mm9", "mm21", "rsi"]])[0]

        mm9 = teste["mm9"].iloc[0]
        mm21 = teste["mm21"].iloc[0]
        rsi = teste["rsi"].iloc[0]

        if pred == 1 and mm9 > mm21 and rsi < 65:
            sinal = "COMPRA"
        elif pred == 0 and mm9 < mm21 and rsi > 35:
            sinal = "VENDA"
        else:
            sinal = "AGUARDAR"

        preco_entrada = df["Close"].iloc[i]
        preco_saida = df["Close"].iloc[i + 2]
        retorno = (preco_saida / preco_entrada) - 1

        if sinal == "COMPRA":
            acerto = 1 if retorno > 0 else 0
            retorno_estrategia = retorno
        elif sinal == "VENDA":
            acerto = 1 if retorno <= 0 else 0
            retorno_estrategia = -retorno
        else:
            acerto = None
            retorno_estrategia = 0

        resultados.append({
            "sinal": sinal,
            "retorno": retorno_estrategia,
            "acerto": acerto,
        })

    bt = pd.DataFrame(resultados)
    operacoes = bt[bt["sinal"] != "AGUARDAR"].copy()

    if operacoes.empty:
        return {
            "operacoes": 0,
            "taxa_acerto": 0,
            "retorno_total": 0,
        }

    taxa_acerto = operacoes["acerto"].mean() * 100
    retorno_total = operacoes["retorno"].sum() * 100

    return {
        "operacoes": len(operacoes),
        "taxa_acerto": round(taxa_acerto, 2),
        "retorno_total": round(retorno_total, 2),
    }


# =========================
# SCORE DO SCANNER
# =========================
def calcular_score(sinal: str, bt: dict, df: pd.DataFrame) -> float:
    if bt is None or df is None or df.empty:
        return 0

    score = 0

    close = df["Close"].iloc[-1]
    mm9 = df["mm9"].iloc[-1]
    mm21 = df["mm21"].iloc[-1]
    mm50 = df["mm50"].iloc[-1]
    rsi = df["rsi"].iloc[-1]
    macd = df["macd"].iloc[-1]
    macd_signal = df["macd_signal"].iloc[-1]

    if sinal == "COMPRA":
        score += 40
    elif sinal == "AGUARDAR":
        score += 5

    if close > mm21:
        score += 10
    if mm9 > mm21:
        score += 10
    if mm21 > mm50:
        score += 10
    if 50 <= rsi <= 65:
        score += 10
    if macd > macd_signal:
        score += 10

    score += min(bt["taxa_acerto"] * 0.2, 20)
    score += min(bt["retorno_total"] * 0.05, 10)

    return round(score, 2)


# =========================
# SCANNER
# =========================
def scanner_oportunidades(lista_ativos, mercado_nome: str, top_n: int = 10):
    oportunidades = []

    for ativo in lista_ativos:
        try:
            df = baixar_dados(ativo)
            if df is None:
                continue

            df = preparar_dados(df)
            if df.empty:
                continue

            resultado_ia = gerar_sinal(df)
            sinal = resultado_ia["sinal"]
            prob_alta = resultado_ia["prob_alta"]
            prob_baixa = resultado_ia["prob_baixa"]
            confianca = resultado_ia["confianca"]

            bt = backtest(df)
            if bt is None:
                continue

            if not (sinal == "COMPRA" and confianca >= 60 and bt["taxa_acerto"] > 55):
                continue

            score = calcular_score(sinal, bt, df)
            preco = round(float(df["Close"].iloc[-1]), 2)

            if len(df) > 1:
                variacao = round(((df["Close"].iloc[-1] / df["Close"].iloc[-2]) - 1) * 100, 2)
            else:
                variacao = 0

            oportunidades.append({
                "mercado": mercado_nome,
                "ativo": ativo,
                "sinal": sinal,
                "score": score,
                "preco": preco,
                "variacao": variacao,
                "prob_alta": prob_alta,
                "prob_baixa": prob_baixa,
                "confianca": confianca,
                "acerto": bt["taxa_acerto"],
                "operacoes": bt["operacoes"],
                "retorno": bt["retorno_total"],
            })

        except Exception as e:
            print(f"Erro no ativo {ativo}: {e}")
            continue

    oportunidades = sorted(
        oportunidades,
        key=lambda x: (x["score"], x["retorno"], x["acerto"]),
        reverse=True,
    )

    return oportunidades[:top_n]


# =========================
# EXECUÇÃO
# =========================
def salvar_historico(ativo, preco, sinal, prob_alta, prob_baixa, confianca, bt) -> None:
    novo = pd.DataFrame([{
        "data_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ativo": ativo,
        "preco": preco,
        "sinal": sinal,
        "prob_alta": prob_alta,
        "prob_baixa": prob_baixa,
        "confianca": confianca,
        "acerto": bt["taxa_acerto"] if bt else 0,
        "operacoes": bt["operacoes"] if bt else 0,
        "retorno": bt["retorno_total"] if bt else 0,
    }])

    if os.path.exists(ARQ_HISTORICO):
        historico = pd.read_csv(ARQ_HISTORICO)
        historico = pd.concat([historico, novo], ignore_index=True)
    else:
        historico = novo

    historico.to_csv(ARQ_HISTORICO, index=False)


def main() -> None:
    linhas = ["SINAIS DOS ATIVOS FIXOS\n"]

    resultados_fixos = []
    resultados_scan_eua = []
    resultados_scan_br = []

    for ativo in ativos:
        try:
            df = baixar_dados(ativo)
            if df is None:
                linha = f"{ativo}: erro ao baixar dados"
                print(linha)
                linhas.append(linha)
                continue

            df = preparar_dados(df)
            if df.empty:
                linha = f"{ativo}: dados insuficientes após preparação"
                print(linha)
                linhas.append(linha)
                continue

            resultado_ia = gerar_sinal(df)
            sinal = resultado_ia["sinal"]
            prob_alta = resultado_ia["prob_alta"]
            prob_baixa = resultado_ia["prob_baixa"]
            confianca = resultado_ia["confianca"]

            bt = backtest(df)
            preco = round(float(df["Close"].iloc[-1]), 2)

            if len(df) > 1:
                variacao = round(((df["Close"].iloc[-1] / df["Close"].iloc[-2]) - 1) * 100, 2)
            else:
                variacao = 0

            variacao_str = f"+{variacao}%" if variacao > 0 else f"{variacao}%"

            taxa_acerto = bt["taxa_acerto"] if bt else 0
            operacoes = bt["operacoes"] if bt else 0
            retorno_total = bt["retorno_total"] if bt else 0

            linha = (
                f"{ativo}: {sinal} | "
                f"Preço: {preco} | "
                f"Variação: {variacao_str} | "
                f"Confiança: {confianca}% | "
                f"Acerto: {taxa_acerto}% | "
                f"Retorno: {retorno_total}%"
            )

            print(linha)
            linhas.append(linha)

            resultados_fixos.append({
                "Ativo": ativo,
                "Sinal": sinal,
                "Preco": preco,
                "Prob_Alta": prob_alta,
                "Prob_Baixa": prob_baixa,
                "Confianca_IA": confianca,
                "Acerto": taxa_acerto,
                "Operacoes": operacoes,
                "Retorno": retorno_total,
            })

            salvar_historico(ativo, preco, sinal, prob_alta, prob_baixa, confianca, bt)

        except Exception as e:
            linha = f"{ativo}: erro -> {e}"
            print(linha)
            linhas.append(linha)

    linhas.append("\nTOP 10 OPORTUNIDADES - SCANNER EUA\n")
    top_eua = scanner_oportunidades(ativos_scan_eua, "EUA", top_n=10)

    if top_eua:
        for i, op in enumerate(top_eua, start=1):
            linha = (
                f"{i}. {op['ativo']} | {op['sinal']} | "
                f"Preço: {op['preco']} | "
                f"Score: {op['score']} | "
                f"Acerto: {op['acerto']}% | "
                f"Retorno: {op['retorno']}%"
            )
            print(linha)
            linhas.append(linha)

            resultados_scan_eua.append({
                "Ativo": op["ativo"],
                "Sinal": op["sinal"],
                "Preco": op["preco"],
                "Score": op["score"],
                "Prob_Alta": op["prob_alta"],
                "Prob_Baixa": op["prob_baixa"],
                "Confianca_IA": op["confianca"],
                "Acerto": op["acerto"],
                "Operacoes": op["operacoes"],
                "Retorno": op["retorno"],
            })
    else:
        linhas.append("Nenhuma oportunidade forte encontrada no scanner EUA.")

    linhas.append("\nTOP 10 OPORTUNIDADES - SCANNER BRASIL\n")
    top_br = scanner_oportunidades(ativos_scan_br, "BRASIL", top_n=10)

    if top_br:
        for i, op in enumerate(top_br, start=1):
            linha = (
                f"{i}. {op['ativo']} | {op['sinal']} | "
                f"Preço: {op['preco']} | "
                f"Score: {op['score']} | "
                f"Acerto: {op['acerto']}% | "
                f"Retorno: {op['retorno']}%"
            )
            print(linha)
            linhas.append(linha)

            resultados_scan_br.append({
                "Ativo": op["ativo"],
                "Sinal": op["sinal"],
                "Preco": op["preco"],
                "Score": op["score"],
                "Prob_Alta": op["prob_alta"],
                "Prob_Baixa": op["prob_baixa"],
                "Confianca_IA": op["confianca"],
                "Acerto": op["acerto"],
                "Operacoes": op["operacoes"],
                "Retorno": op["retorno"],
            })
    else:
        linhas.append("Nenhuma oportunidade forte encontrada no scanner Brasil.")

    texto = "\n".join(linhas)

    colunas_scanner = [
        "Ativo", "Sinal", "Preco", "Score",
        "Prob_Alta", "Prob_Baixa", "Confianca_IA",
        "Acerto", "Operacoes", "Retorno",
    ]

    pd.DataFrame(resultados_fixos).to_csv(ARQ_FIXOS, index=False)

    df_eua = pd.DataFrame(resultados_scan_eua)
    if df_eua.empty:
        df_eua = pd.DataFrame(columns=colunas_scanner)
    else:
        df_eua = df_eua[colunas_scanner]
    df_eua.to_csv(ARQ_EUA, index=False)

    df_br = pd.DataFrame(resultados_scan_br)
    if df_br.empty:
        df_br = pd.DataFrame(columns=colunas_scanner)
    else:
        df_br = df_br[colunas_scanner]
    df_br.to_csv(ARQ_BR, index=False)

    if EMAIL_REMETENTE and EMAIL_SENHA_APP and EMAIL_DESTINO:
        enviar_email(texto)
        print("\nE-mail enviado com sucesso.")
    else:
        print("\nVariáveis de e-mail não configuradas. E-mail não enviado.")


if __name__ == "__main__":
    main()
