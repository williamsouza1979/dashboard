# We'll create a Streamlit dashboard project with app.py, requirements.txt, and README.md,
# then zip them for the user to download.

import os, textwrap, json, zipfile, io, pandas as pd

project_dir = "/mnt/data/streamlit_dashboard"
os.makedirs(project_dir, exist_ok=True)

app_py = r'''# app.py
# -*- coding: utf-8 -*-
import io
import json
import base64
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# ---------------------------
# Configuração de página
# ---------------------------
st.set_page_config(page_title="Dashboard CSV - Análises Rápidas", layout="wide")

# ---------------------------
# Utilidades
# ---------------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _auto_cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Tenta converter colunas de texto com números (com vírgula/ponto) para numérico."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().astype(str).head(50).str.strip()
            # Heurística: se a maioria das amostras parecer número, converte
            looks_numeric = (sample.str.replace('.', '', regex=False)
                                   .str.replace(',', '', regex=False)
                                   .str.replace('-', '', regex=False)
                                   .str.replace(' ', '', regex=False)
                                   .str.match(r'^\d+$', na=False)).mean() > 0.6
            # Outra heurística: presença consistente de dígitos e separadores
            has_digits = sample.str.contains(r'\d', regex=True).mean() > 0.6
            if looks_numeric or has_digits:
                # Tenta primeiro decimal=',' (pt-BR)
                c = (df[col].astype(str).str.replace('.', '', regex=False)
                                  .str.replace(',', '.', regex=False))
                coerced = pd.to_numeric(c, errors='coerce')
                # Se converter pouco, tenta direto to_numeric padrão
                if coerced.notna().mean() < 0.3:
                    coerced = pd.to_numeric(df[col], errors='coerce')
                # Aplica conversão se houver ganho
                if coerced.notna().sum() >= df[col].notna().sum() * 0.3:
                    df[col] = coerced
    return df

def _read_csv_any(file, sep_choice, encoding_choice):
    """Lê CSV suportando diferentes separadores/encodings; retorna (df, msg)."""
    # Tenta com a seleção do usuário primeiro
    tried = []
    errors = []
    seps = [sep_choice] if sep_choice else [',',';','\t','|']
    encs = [encoding_choice] if encoding_choice else ['utf-8','latin-1','utf-16']
    for s in seps:
        for e in encs:
            try:
                df = pd.read_csv(file, sep=s, encoding=e, low_memory=False)
                return df, f"Lido com separador '{s}' e encoding '{e}'."
            except Exception as ex:
                tried.append((s,e))
                errors.append(str(ex))
                file.seek(0)
    return None, f"Falha ao ler CSV. Tentativas: {tried[:4]}..."

def _concat_align(current: pd.DataFrame, new: pd.DataFrame, mode: str) -> pd.DataFrame:
    new = _normalize_columns(new)
    new = _auto_cast_numeric(new)
    if current is None or mode == "Substituir":
        return new
    # Alinhar colunas
    return pd.concat([current, new], ignore_index=True, sort=False)

def _desc_stats(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=np.number)
    if numeric.empty:
        return pd.DataFrame({"info": ["Nenhuma coluna numérica encontrada."]})
    agg = numeric.agg(['sum','mean','min','max','count']).T
    agg = agg.rename(columns={'sum':'soma','mean':'média','min':'mínimo','max':'máximo','count':'contagem'})
    agg.index.name = 'coluna'
    return agg.reset_index()

def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

def _to_json_bytes(df: pd.DataFrame) -> bytes:
    return df.to_json(orient='records', force_ascii=False, indent=2).encode('utf-8')

# ---------------------------
# Estado
# ---------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "last_validation" not in st.session_state:
    st.session_state.last_validation = ""

# ---------------------------
# Checklist inicial
# ---------------------------
with st.expander("✅ Checklist antes de começar (clique para ver)", expanded=True):
    st.markdown("""
1. Confirme o formato dos dados (CSV) e a codificação (UTF-8/Latin-1).
2. Verifique se há cabeçalho na primeira linha e nomes de colunas claros.
3. Separe os arquivos que deseja **adicionar** ou **substituir**.
4. Defina quais colunas serão usadas nos gráficos e na tabela dinâmica.
5. Escolha como deseja **exportar** (CSV e/ou JSON) os resultados.
""")

st.title("📊 Dashboard de Análise de CSV (Streamlit)")

# ---------------------------
# Importação de arquivos
# ---------------------------
st.sidebar.header("1) Importar CSV(s)")
mode = st.sidebar.radio("Modo de atualização de dados:", ["Substituir", "Adicionar"], index=0)
sep_choice = st.sidebar.selectbox("Separador (opcional)", ["(auto)", ",", ";", "\\t", "|"], index=0)
encoding_choice = st.sidebar.selectbox("Encoding (opcional)", ["(auto)", "utf-8", "latin-1", "utf-16"], index=0)
if sep_choice == "(auto)": sep_choice = None
if encoding_choice == "(auto)": encoding_choice = None

uploaded = st.sidebar.file_uploader("Selecione um ou mais arquivos CSV", type=["csv"], accept_multiple_files=True)

if uploaded:
    built = None
    logs = []
    for up in uploaded:
        df, msg = _read_csv_any(up, sep_choice, encoding_choice)
        logs.append(msg)
        if df is not None:
            before = 0 if st.session_state.df is None else len(st.session_state.df)
            built = _concat_align(st.session_state.df, df, mode)
            after = len(built)
            st.session_state.df = built
            logs.append(f"Arquivo '{up.name}' processado ({len(df)} linhas). Total atual: {after} linhas.")
        else:
            logs.append(f"⚠️ Não foi possível ler '{up.name}'.")

    # Validação após importação
    if st.session_state.df is not None and not st.session_state.df.empty:
        st.success(f"Importação concluída. Linhas: {len(st.session_state.df)}, Colunas: {st.session_state.df.shape[1]}.")
        st.caption("Validação: dados presentes e colunas normalizadas. Prosseguindo para análises.")
    else:
        st.error("Falha: não há dados carregados. Verifique separador/encoding e tente novamente.")
        st.caption("Auto-correção sugerida: tente outro separador/encoding ou confirme se o arquivo possui cabeçalho.")
    with st.expander("Detalhes da importação"):
        for l in logs: st.write("•", l)

# ---------------------------
# Visualização & Análises
# ---------------------------
df = st.session_state.df
if df is not None and not df.empty:
    st.subheader("📄 Prévia da Tabela")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("📌 Estatísticas Descritivas (colunas numéricas)")
    stats = _desc_stats(df)
    st.dataframe(stats, use_container_width=True)
    st.caption("Validação: estatísticas calculadas para colunas numéricas.")

    st.subheader("📈 Gráficos")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols = df.columns.tolist()

    tab_bar, tab_line, tab_pie = st.tabs(["Barras", "Linhas", "Pizza"])

    with tab_bar:
        x = st.selectbox("Eixo X (categoria ou numérica):", all_cols, key="bar_x")
        y = st.selectbox("Eixo Y (numérica):", num_cols, key="bar_y")
        agg = st.selectbox("Agregação:", ["sum", "mean", "min", "max", "count"], key="bar_agg")
        if x and y:
            df_plot = df.groupby(x, dropna=False)[y].agg(agg).reset_index()
            fig = px.bar(df_plot, x=x, y=y, title=f"Barra de {agg}({y}) por {x}")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Validação: gráfico de barras gerado com sucesso.")

    with tab_line:
        x = st.selectbox("Eixo X (sequência/tempo ou categoria):", all_cols, key="line_x")
        y = st.multiselect("Eixos Y (numéricas):", num_cols, default=num_cols[:1], key="line_y")
        if x and y:
            df_plot = df[[x]+y].dropna()
            fig = px.line(df_plot, x=x, y=y, title=f"Linhas para {', '.join(y)} ao longo de {x}")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Validação: gráfico de linhas gerado com sucesso.")

    with tab_pie:
        names = st.selectbox("Categorias (rótulos):", all_cols, key="pie_names")
        values = st.selectbox("Valores (numérica):", num_cols, key="pie_values")
        if names and values:
            df_plot = df.groupby(names, dropna=False)[values].sum().reset_index()
            fig = px.pie(df_plot, names=names, values=values, title=f"Pizza de {values} por {names}")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Validação: gráfico de pizza gerado com sucesso.")

    st.subheader("🧮 Tabela Dinâmica (Pivot)")
    with st.container():
        idx_cols = st.multiselect("Linhas (índices):", options=all_cols, key="pivot_idx")
        col_cols = st.multiselect("Colunas (opcional):", options=all_cols, key="pivot_cols")
        val_col = st.selectbox("Valor (numérica):", options=num_cols, key="pivot_val")
        aggfunc = st.selectbox("Função:", ["sum","mean","min","max","count"], key="pivot_agg", index=0)

        pivot_df = None
        if val_col:
            try:
                pivot_df = pd.pivot_table(df, index=idx_cols if idx_cols else None,
                                          columns=col_cols if col_cols else None,
                                          values=val_col, aggfunc=aggfunc, fill_value=0).reset_index()
                st.dataframe(pivot_df, use_container_width=True)
                st.caption("Validação: pivot gerado com sucesso.")
            except Exception as ex:
                st.error(f"Erro ao gerar pivot: {ex}")
                st.caption("Auto-correção sugerida: selecione colunas válidas e uma métrica numérica.")

    # ---------------------------
    # Exportação
    # ---------------------------
    st.subheader("📤 Exportar")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.download_button("Baixar Dados (CSV)", data=_to_csv_bytes(df), file_name="dados.csv", mime="text/csv")
    with col2:
        st.download_button("Baixar Dados (JSON)", data=_to_json_bytes(df), file_name="dados.json", mime="application/json")
    with col3:
        st.download_button("Baixar Estatísticas (CSV)", data=_to_csv_bytes(stats), file_name="estatisticas.csv", mime="text/csv")
    with col4:
        if 'pivot_df' in locals() and pivot_df is not None:
            st.download_button("Baixar Pivot (CSV)", data=_to_csv_bytes(pivot_df), file_name="pivot.csv", mime="text/csv")
        else:
            st.info("Gere a pivot para habilitar o download.")

    st.caption("Validação: botões de exportação ativos. Baixe e verifique os arquivos gerados.")

# ---------------------------
# Rodapé / ajuda
# ---------------------------
st.markdown("---")
st.markdown("**Dica:** para atualizar o dashboard com novos CSVs, basta enviar novos arquivos na barra lateral em modo *Adicionar* ou *Substituir*. Todas as visualizações serão recalculadas automaticamente.")
st.markdown("Feito com ❤️ usando Streamlit + Pandas + Plotly.")
'''

requirements_txt = """streamlit==1.36.0
pandas>=2.1.0
plotly>=5.22.0
numpy>=1.26.0
pyarrow>=15.0.0
"""

readme_md = r'''# Dashboard de Análise de CSV (Streamlit)

Este projeto cria um dashboard simples e gratuito para analisar arquivos CSV com **gráficos (barras, linhas, pizza)**, **tabelas dinâmicas (pivot)** e **estatísticas descritivas**. Permite **importar múltiplos CSVs**, **adicionar ou substituir** dados e **exportar** resultados em **CSV/JSON**.

## Como executar localmente
```bash
# 1) Crie um virtualenv (opcional)
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 2) Instale dependências
pip install -r requirements.txt

# 3) Rode o app
streamlit run app.py
