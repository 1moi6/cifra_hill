import streamlit as st
import numpy as np
from sympy import Matrix
import json
import string
from io import StringIO
import pandas as pd


if "K" not in st.session_state:
    st.session_state.K = None
if "K_name" not in st.session_state:
    st.session_state.K_name = None

if "blocos" not in st.session_state:
    st.session_state.blocos = None
if "enumeracao" not in st.session_state:
    st.session_state.enumeracao = None
if "produto" not in st.session_state:
    st.session_state.produto = None
if "mapeamento" not in st.session_state:
    st.session_state.mapeamento = None
if "inverso" not in st.session_state:
    st.session_state.inverso = None

# ======== NOVO: helpers p/ grid de alfabeto ========
def alphabet_to_grid(alpha: str, cols: int = 16) -> pd.DataFrame:
    cells = list(alpha)
    # completa para múltiplo de cols com strings vazias (apenas UI)
    pad = (-len(cells)) % cols
    cells += [""] * pad
    rows = [cells[i:i+cols] for i in range(0, len(cells), cols)]
    df = pd.DataFrame(rows)
    return df

def grid_to_alphabet(df: pd.DataFrame) -> str:
    # concatena linha a linha, ignora vazios e garante um char por célula
    chars = []
    for _, row in df.iterrows():
        for c in row.tolist():
            if isinstance(c, str) and len(c) > 0:
                chars.append(c[0])  # pega só o 1º caractere
    # remove duplicatas preservando ordem
    seen = set(); out = []
    for ch in chars:
        if ch not in seen:
            seen.add(ch); out.append(ch)
    # garante espaço ao final
    if " " not in out:
        out.append(" ")
    return "".join(out)



# =========================
# Utilidades da Cifra de Hill
# =========================

def validate_alphabet(alpha: str):
    # Garante unicidade e não vazio
    if len(alpha) == 0:
        raise ValueError("O alfabeto não pode ser vazio.")
    if len(set(alpha)) != len(alpha):
        # remove duplicatas mantendo ordem
        seen = set()
        dedup = []
        for ch in alpha:
            if ch not in seen:
                dedup.append(ch)
                seen.add(ch)
        alpha = "".join(dedup)
    return alpha

def build_maps(alpha: str):
    mapeamento = {c: i for i, c in enumerate(alpha)}
    inverso = {i: c for c, i in mapeamento.items()}
    st.session_state.mapeamento = mapeamento
    st.session_state.inverso = inverso
    return mapeamento, inverso

def check_text_in_alphabet(texto: str, alpha: str):
    missing = sorted({c for c in texto if c not in alpha})
    return missing

def pad_text_blocks(text: str, n: int, mode: str):
    """
    mode: 'full' (texto inteiro) ou 'words' (por palavras)
    padding: espaços ' ' até múltiplo de n
    """
    if mode == "full":
        pad_len = (-len(text)) % n
        return text + (" " * pad_len)
    else:
        # por palavras: split em whitespaces preservando separadores
        # Estratégia: split nos espaços simples ' ' mantendo múltiplos espaços como separadores
        # Para simplificar, usamos split padrão por whitespace e depois join com único espaço.
        # (Se quiser preservação exata de espaços, podemos mudar para uma tokenização mais fina.)
        words = text.split()
        padded_words = []
        for w in words:
            pad_len = (-len(w)) % n
            padded_words.append(w + (" " * pad_len))
        # Mantém separadores simples (um espaço)
        return " ".join(padded_words)

def text_to_numbers(text: str, mapeamento: dict):
    return [mapeamento[c] for c in text]

def numbers_to_text(nums, inverso: dict):
    return "".join(inverso[i] for i in nums)

def is_invertible_mod(K: np.ndarray, mod: int) -> bool:
    try:
        MK = Matrix(K)
        det = int(MK.det()) % mod
        if det == 0:
            return False
        # testa inversa modular
        _ = MK.inv_mod(mod)
        return True
    except Exception:
        return False

def inv_mod_matrix(K: np.ndarray, mod: int) -> np.ndarray:
    MK = Matrix(K)
    invK = MK.inv_mod(mod)
    # sympy -> numpy int
    invK = np.array(invK.tolist(), dtype=int)
    return invK

def hill_encrypt(text: str, K: np.ndarray, alpha: str, mode: str):
    m = len(alpha)
    n = K.shape[0]
    mapeamento, inverso = build_maps(alpha)

    # padding
    padded = pad_text_blocks(text, n, mode)
    nums = text_to_numbers(padded, mapeamento)
    st.session_state.enumeracao = {b:a for a,b in zip(nums,padded)}

    out = []
    blocos = []
    produto = []
    for i in range(0, len(nums), n):
        bloco = np.array(nums[i:i+n], dtype=int)
        blocos.append([int(v) for v in bloco])
        res = K.dot(bloco) % m
        produto.append(res)
        out.extend(res.tolist())
    
    st.session_state.blocos={f"bloco {i+1}":[(inverso[x], x) for x in b] for i,b in enumerate(blocos)}
    st.session_state.produto = produto
    return numbers_to_text(out, inverso), padded

def hill_decrypt(text: str, K: np.ndarray, alpha: str, mode: str):
    m = len(alpha)
    n = K.shape[0]
    mapeamento, inverso = build_maps(alpha)
    invK = inv_mod_matrix(K, m)

    # exige comprimento múltiplo de n
    if len(text) % n != 0:
        # completa com espaços para poder decifrar
        text = text + (" " * ((-len(text)) % n))

    nums = text_to_numbers(text, mapeamento)
    out = []
    for i in range(0, len(nums), n):
        bloco = np.array(nums[i:i+n], dtype=int)
        res = invK.dot(bloco) % m
        out.extend(res.tolist())
    dec = numbers_to_text(out, inverso)
    # Remoção opcional de padding fica a critério do usuário (checkbox na UI)
    return dec

def parse_matrix_from_textarea(txt: str):
    """
    Aceita formatos:
    - JSON array: [[...],[...],...]
    - CSV-like simples: linhas por quebra de linha e elementos separados por espaço ou vírgula
    """
    txt = txt.strip()
    if not txt:
        raise ValueError("A área de texto da matriz está vazia.")

    # Tenta JSON primeiro
    try:
        data = json.loads(txt)
        arr = np.array(data, dtype=int)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("A matriz precisa ser quadrada.")
        return arr
    except Exception:
        pass

    # Tenta CSV-like
    rows = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        # separa por vírgula ou espaço
        if "," in line:
            parts = [p.strip() for p in line.split(",") if p.strip()]
        else:
            parts = [p for p in line.split() if p]
        rows.append([int(x) for x in parts])

    arr = np.array(rows, dtype=int)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("A matriz precisa ser quadrada.")
    return arr

def load_matrices_from_json(file_bytes: bytes):
    data = json.loads(file_bytes.decode("utf-8"))
    if not isinstance(data, dict) or "matrices" not in data:
        raise ValueError('JSON inválido. Esperado objeto com chave "matrices".')
    items = data["matrices"]
    parsed = []
    for it in items:
        name = it.get("name", "sem_nome")
        mat = np.array(it.get("matrix", []), dtype=int)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            continue
        parsed.append((name, mat))
    if not parsed:
        raise ValueError("Nenhuma matriz quadrada válida encontrada no JSON.")
    return parsed

def generate_random_invertible(n: int, mod: int, max_tries: int = 5000):
    rng = np.random.default_rng()
    for _ in range(max_tries):
        K = rng.integers(0, mod, size=(n, n))
        if is_invertible_mod(K, mod):
            return K
    raise RuntimeError("Falha ao gerar matriz invertível módulo M após várias tentativas.")


# =========================
# UI (Streamlit)
# =========================

st.set_page_config(page_title="Aplicativo para Cifra de Hill", layout="wide")
st.title("🔐 Aplicativo Para Cifra de Hill")

# ======== NOVO: estado inicial para grade do alfabeto ========
# garante que sempre temos um alfabeto no estado
if "alpha" not in st.session_state:
    st.session_state.alpha = string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation + " "
    # força inclusão do espaço
    if " " not in st.session_state.alpha:
        st.session_state.alpha += " "
    st.session_state.M = len(st.session_state.alpha)

# número de colunas da grade
if "alpha_grid_cols" not in st.session_state:
    st.session_state.alpha_grid_cols = 20

# dataframe inicial da grade
if "alpha_grid_df" not in st.session_state:
    st.session_state.alpha_grid_df = alphabet_to_grid(
        st.session_state.alpha,
        st.session_state.alpha_grid_cols
    )

tab_tool, tab_info, tab_etapas = st.tabs(["Codificação e Decodificação de Texto",'Alfabeto e Numeração', "Detalhes da Codificação"])

with st.sidebar:
    st.header("Configurações")

    # Texto de entrada e operação
    oper = st.radio("Operação:", ["Codificar", "Decodificar"], index=0)
    mode = st.radio("Modo de processamento:", ["Texto inteiro", "Por palavras"], index=0)
    mode_key = "full" if mode == "Texto inteiro" else "words"

    # Alfabeto
    alpha = st.session_state.alpha
    try:
        alpha = validate_alphabet(alpha)
    except Exception as e:
        st.error(f"Erro no alfabeto: {e}")

    # Opções de tratamento de quebras de linha
    newline_as_space = st.checkbox("Quebras de linha como espaço", value=True)
    strip_trailing = st.checkbox("Remover espaços de ajuste", value=False,
                                    help="Útil após decodificar.")

    # Dimensão da matriz
    n = st.number_input("Dimensão da matriz (n×n)", min_value=2, max_value=12, value=5, step=1)

    # Matriz-chave: origem
    st.subheader("Matriz-chave (K)")
    key_source = st.selectbox(
        "Origem da matriz",
        ["Gerar aleatória", "Inserir manualmente", "Carregar de arquivo (JSON)"],
        index=0
    )

    K = None
    loaded_choices = []
    chosen_loaded = None

    M = len(alpha)

    if key_source == "Gerar aleatória":
        if st.button("Gerar matriz invertível (mod M)",use_container_width=True):
            try:
                K = generate_random_invertible(n, M)
                st.session_state.K = K
                st.session_state.K_name = f"Random_{n}x{n}_mod{M}"
                st.success("Matriz K gerada!")
                st.write(st.session_state.K)
            except Exception as e:
                st.error(f"Falha ao gerar matriz: {e}")

    elif key_source == "Inserir manualmente":
        st.caption("Cole uma matriz quadrada n×n. Aceita JSON ([[...],[...],...]) ou linhas separadas por vírgula/espaço.")
        txt_mat = st.text_area("Matriz K", height=180, value="")
        if st.button("Usar matriz colada"):
            try:
                K = parse_matrix_from_textarea(txt_mat)
                if K.shape[0] != n:
                    st.warning(f"A matriz colada é {K.shape[0]}×{K.shape[1]}, mas você selecionou n={n}. Ajuste n ou a matriz.")
                if not is_invertible_mod(K, M):
                    st.error("A matriz não é invertível módulo M (det ≡ 0 ou sem inversa).")
                else:
                    st.session_state.K = K
                    st.session_state.K_name = f"Manual_{K.shape[0]}x{K.shape[1]}_mod{M}"
                    st.success("Matriz válida e salva.")
                    st.write(st.session_state.K)
            except Exception as e:
                st.error(f"Erro ao interpretar a matriz: {e}")

    else:  # Carregar de arquivo (JSON)
        up = st.file_uploader("Envie um JSON com matrizes", type=["json"])
        if up is not None:
            try:
                items = load_matrices_from_json(up.read())
                loaded_choices = [f"{name}  ({mat.shape[0]}×{mat.shape[1]})" for name, mat in items]
                chosen_loaded = st.selectbox("Escolha uma matriz do arquivo", loaded_choices)
                if chosen_loaded:
                    idx = loaded_choices.index(chosen_loaded)
                    name, mat = items[idx]
                    K = mat
                    st.write("Matriz selecionada:")
                    st.write(K)
                    if K.shape[0] != n:
                        st.warning(f"A matriz selecionada é {K.shape[0]}×{K.shape[1]}, mas você selecionou n={n}. Ajuste n ou a matriz.")
                    if not is_invertible_mod(K, M):
                        st.error("A matriz selecionada NÃO é invertível mod M.")
                    else:
                        st.session_state.K = mat
                        st.session_state.K_name = f"{name}_mod{M}"
                        st.success("Matriz válida e salva.")
                        st.write(st.session_state.K)
            except Exception as e:
                st.error(f"Erro ao carregar JSON: {e}")

with tab_tool:
    # Entrada do texto
    st.markdown("#### Entre com o seu texto")
    texto = st.text_area("Use esse espaço para inserir o seu texto:", height=180, placeholder="Digite aqui seu texto (use apenas caracteres do alfabeto definido).")

    if newline_as_space and texto:
        texto = texto.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")

    # Botões de ação
    col0,colA, colB = st.columns([0.3,0.4, 0.3])
    with colA:
        executar = st.button(f"{oper} o texto",use_container_width=True)
        

    if executar:
        K = st.session_state.K  # <- pegue do estado
        if not alpha or " " not in alpha:
            st.error("Inclua o espaço ' ' no alfabeto (necessário para o padding).")
        elif not texto:
            st.warning("Forneça um texto.")
        elif K is None:
            st.error("Selecione/forneça uma matriz K válida (invertível modulo M).")
        else:
            # Validação de caracteres
            missing = check_text_in_alphabet(texto, alpha)
            if missing:
                st.error(f"Seu texto contém caracteres fora do alfabeto: {missing}")
            else:
                try:
                    if oper.startswith("Codificar"):
                        cifrado, padded = hill_encrypt(texto, K, alpha, mode_key)
                        out = cifrado
                        st.markdown("#### Seu texto codificado")
                        st.code(out)
                        st.caption(f"Comprimento (entrada/pad): {len(texto)}/{len(padded)}  •  M = {len(alpha)}  •  n = {K.shape[0]}")
                        cols = st.columns([0.3,0.4, 0.3])
                        cols[1].download_button("Baixar texto codificado", data=out, file_name="cifrado.txt", mime="text/plain",use_container_width=True)
                        
                    else:
                        dec = hill_decrypt(texto, K, alpha, mode_key)
                        out = dec.rstrip() if strip_trailing else dec
                        st.code(out)
                        st.caption(f"Comprimento (entrada): {len(texto)}  •  M = {len(alpha)}  •  n = {K.shape[0]}")
                        cols = st.columns([0.3,0.4, 0.3])
                        cols[1].download_button("Baixar texto decodificado", data=out, file_name="decifrado.txt", mime="text/plain",use_container_width=True)

                except Exception as e:
                    st.error(f"Erro durante o processamento: {e}")


with tab_etapas:
    if st.session_state.blocos:
        with st.container(border=True):
            st.markdown(f'#### **Separação em Blocos**')
            for (b,v),c in zip(st.session_state.blocos.items(),st.session_state.produto):
                st.divider()
                st.markdown(f'**{b}**')
                dc = {"Texto":[a[0] for a in v], "Código":[a[1] for a in v]}
                st.code(f"Texto sem codificaçõo: {dc['Texto']}\n\nEnumeração: {dc['Código']}\n\nCódigo: {c}\n\nTexto codificado: {[st.session_state.inverso[x] for x in c]}")
                # st.code(f"\n\nTexto: {c}")


# ======== ABA INFORMAÇÕES ========
with tab_info:
    st.subheader("Alfabeto")
    st.caption("Cada célula deve conter **no máximo 1** caractere. Duplicatas serão removidas, e o espaço `' '` é sempre incluído automaticamente ao final.")
    
    # editor
    edited_alpha_df = st.data_editor(
        st.session_state.alpha_grid_df,
        key="alpha_editor",
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
    )
    mapeamento, inverso = build_maps(st.session_state.alpha)
    apply_cols = st.columns([0.35,0.3,0.35])
    with apply_cols[1]:
        if st.button("Aplicar alterações no alfabeto",use_container_width=True):
            try:
                new_alpha = grid_to_alphabet(edited_alpha_df)
                new_alpha = validate_alphabet(new_alpha)  # mantém dedup + garante espaço
                alpha_changed = (new_alpha != alpha)
                st.session_state.alpha = new_alpha
                st.session_state.alpha_grid_df = edited_alpha_df.copy()
                st.session_state.M = len(new_alpha)
                mapeamento, inverso = build_maps(new_alpha)
               

                # se M mudou, revalidar/limpar K
                K = st.session_state.get("K", None)
                if K is not None and not is_invertible_mod(K, st.session_state.M):
                    st.warning("O alfabeto mudou (M alterado). A matriz K salva deixou de ser invertível. Ela foi descartada.")
                    st.session_state.K = None
                    st.session_state.K_name = None

                st.success(f"Alfabeto atualizado! M = {len(st.session_state.alpha)}")
            except Exception as e:
                st.error(f"Erro ao aplicar alfabeto do grid: {e}")
   
    st.code(list(st.session_state.mapeamento.items())) 
   
    st.markdown("---")
    st.markdown("#### Matrizes de codificação/decodificação")

    M = len(st.session_state.alpha)
    K = st.session_state.get("K", None)

    if K is None:
        st.info("Nenhuma matriz K definida ainda. Gere, cole ou carregue na aba **Ferramenta**.")
    else:
        st.write(f"**Matriz de codificação K**  (n = {K.shape[0]}, módulo M = {M})")
        dfK = pd.DataFrame(K)
        # st.data_editor(dfK, use_container_width=True, hide_index=True, disabled=True)
        st.code(K)
        try:
            invK = inv_mod_matrix(np.array(K, dtype=int), M)
            st.write("**Matris de decodificação K⁻¹ (mod M)**")
            dfInv = pd.DataFrame(invK)
            # st.data_editor(dfInv, use_container_width=True, hide_index=True, disabled=True)
            st.code(invK)
        except Exception as e:
            st.error(f"Não foi possível calcular K⁻¹ (mod M): {e}")


# Rodapé com dicas
st.markdown("---")
st.markdown(
    """
**Dicas:**
- Se o texto tiver caracteres fora do alfabeto, adicione-os ao campo de alfabeto (inclusive `\\n` se quiser usar quebras de linha).
- Para *Por palavras*, cada palavra recebe padding separadamente; os separadores são normalizados para um único espaço.
- Para reprodutibilidade, evite “Gerar aleatória” em produção; salve K num JSON e use como matriz pré-definida.
"""
)
