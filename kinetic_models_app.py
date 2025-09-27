import streamlit as st
import hashlib
from datetime import datetime, timezone
import hmac
import time

# =============================================================================
# ===== INÍCIO DO CÓDIGO DE PROTEÇÃO FINAL ====================================
# =============================================================================
def init_connection():
    """Inicializa conexão com Supabase. Requer secrets configurados."""
    try:
        from st_supabase_connection import SupabaseConnection
        return st.connection("supabase", type=SupabaseConnection)
    except Exception as e:
        st.error(f"Erro ao conectar com Supabase: {e}")
        return None

def verify_and_consume_nonce(token: str) -> tuple[bool, str | None]:
    """Verifica um token de uso único (nonce) no banco de dados e o consome."""
    conn = init_connection()
    if not conn:
        return False, None

    try:
        # 1. Cria o hash do token recebido para procurar no banco
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        # 2. Procura pelo token no banco de dados
        response = conn.table("auth_tokens").select("*").eq("token_hash", token_hash).execute()
        
        if not response.data:
            st.error("Token de acesso inválido ou não encontrado.")
            return False, None
        
        token_data = response.data[0]
        
        # 3. Verifica se o token já foi utilizado
        if token_data["is_used"]:
            st.error("Este link de acesso já foi utilizado e não é mais válido.")
            return False, None
            
        # 4. Verifica se o token expirou
        expires_at = datetime.fromisoformat(token_data["expires_at"])
        if datetime.now(timezone.utc) > expires_at:
            st.error("O link de acesso expirou. Por favor, gere um novo no portal.")
            return False, None
            
        # 5. Se tudo estiver correto, marca o token como usado (consumido)
        conn.table("auth_tokens").update({"is_used": True}).eq("id", token_data["id"]).execute()
        
        user_email = token_data["user_email"]
        return True, user_email
        
    except Exception as e:
        st.error(f"Ocorreu um erro crítico durante a validação do acesso: {e}")
        return False, None

def verify_auth_token(token: str, secret_key: str) -> tuple:
    """Verifica um token de autenticação HMAC-SHA256 com timestamp."""
    try:
        parts = token.split(':')
        if len(parts) != 3:
            return False, None
        
        email, timestamp, signature = parts
        
        # 1. Verifica se o token expirou (validade de 1 hora)
        if int(time.time()) - int(timestamp) > 3600:
            st.error("Token de autenticação expirado.")
            return False, None
        
        # 2. Recria a assinatura esperada para verificação
        message = f"{email}:{timestamp}"
        expected_signature = hmac.new(secret_key.encode(), message.encode(), hashlib.sha256).hexdigest()
        
        # 3. Compara as assinaturas de forma segura
        if hmac.compare_digest(signature, expected_signature):
            return True, email
        else:
            st.error("Token de autenticação inválido.")
            return False, None
            
    except Exception as e:
        st.error(f"Erro ao verificar token: {e}")
        return False, None

def check_authentication():
    """Verifica autenticação usando sistema duplo: Supabase + HMAC fallback"""
    # --- Lógica Principal de Autenticação ---
    query_params = st.query_params
    access_token = query_params.get("access_token")
    auth_token = query_params.get("auth_token")

    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    # Primeiro tenta Supabase (access_token)
    if access_token and not st.session_state.authenticated:
        time.sleep(1)  # PAUSA ESTRATÉGICA PARA EVITAR RACE CONDITION
        is_valid, email = verify_and_consume_nonce(access_token)
        if is_valid:
            st.session_state.authenticated = True
            st.session_state.user_email = email
            return True

    # Fallback para HMAC (auth_token)
    if auth_token and not st.session_state.authenticated:
        try:
            auth_secrets = st.secrets.get("auth", {})
            secret_key = auth_secrets.get("token_secret_key")

            if secret_key:
                is_valid, email = verify_auth_token(auth_token, secret_key)
                if is_valid:
                    st.session_state.authenticated = True
                    st.session_state.user_email = email
                    return True
            else:
                st.error("Chave secreta de autenticação não configurada no aplicativo.")
                
        except Exception as e:
            st.error(f"Ocorreu um erro durante a autenticação: {e}")

    return st.session_state.get('authenticated', False)

def show_access_denied():
    """Mostra página de acesso negado"""
    st.title("🔒 Acesso Restrito")
    st.error("Este aplicativo requer autenticação. Por favor, faça o login através do portal.")
    
    st.link_button(
        "Ir para o Portal de Login",
        "https://huggingface.co/spaces/MarioNunes34/Portal",
        use_container_width=True,
        type="primary"
    )
    st.stop()

# =============================================================================
# ===== FIM DO CÓDIGO DE PROTEÇÃO =============================================
# =============================================================================

# Verificação de autenticação
if not check_authentication():
    show_access_denied()

# Mensagem de boas-vindas para o usuário autenticado
st.success(f"Autenticação bem-sucedida! Bem-vindo, {st.session_state.get('user_email', 'usuário')}.")

# =============================================================================
# ===== INÍCIO DO APLICATIVO PRINCIPAL ======================================
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import io
from datetime import datetime

# Configuração da página
st.set_page_config(page_title="Ajuste de Modelos Cinéticos", layout="wide")

# Título
st.title("Ajuste de Modelos Cinéticos")
st.markdown("**Modelos disponíveis:** Pseudo-Second Order, Intraparticle Diffusion, Elovich")

# Definição dos modelos cinéticos
def pseudo_second_order(t, k2, qe):
    """Modelo Pseudo-Segunda Ordem: qt = (k2 * qe² * t) / (1 + k2 * qe * t)"""
    return (k2 * qe**2 * t) / (1 + k2 * qe * t)

def intraparticle_diffusion(t, kp, C):
    """Modelo de Difusão Intrapartícula: qt = kp * sqrt(t) + C"""
    return kp * np.sqrt(t) + C

def elovich(t, alpha, beta):
    """Modelo de Elovich: qt = (1/beta) * ln(alpha * beta) + (1/beta) * ln(t)"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (1/beta) * np.log(alpha * beta) + (1/beta) * np.log(t)
        # Substituir valores inválidos por 0
        result = np.where(np.isfinite(result), result, 0)
    return result

# Função para gerar curvas suaves
def generate_smooth_curve(model_name, params, t_data, n_points=200):
    """Gera curva suave para o modelo especificado"""
    t_smooth = np.linspace(np.min(t_data), np.max(t_data), n_points)
    
    if model_name == 'Pseudo-Second Order':
        qt_smooth = pseudo_second_order(t_smooth, params['k2'], params['qe'])
    elif model_name == 'Intraparticle Diffusion':
        qt_smooth = intraparticle_diffusion(t_smooth, params['kp'], params['C'])
    elif model_name == 'Elovich':
        qt_smooth = elovich(t_smooth, params['alpha'], params['beta'])
    else:
        return None, None
    
    return t_smooth, qt_smooth

# Função para calcular R²
def r_squared(y_observed, y_predicted):
    ss_res = np.sum((y_observed - y_predicted) ** 2)
    ss_tot = np.sum((y_observed - np.mean(y_observed)) ** 2)
    return 1 - (ss_res / ss_tot)

# Função para calcular R² ponderado
def weighted_r_squared(y_observed, y_predicted, weights):
    """Calcula R² ponderado considerando os pesos (inverso da variância)"""
    mean_obs = np.average(y_observed, weights=weights)
    ss_res = np.sum(weights * (y_observed - y_predicted) ** 2)
    ss_tot = np.sum(weights * (y_observed - mean_obs) ** 2)
    return 1 - (ss_res / ss_tot)

# Função para calcular chi-quadrado reduzido
def reduced_chi_squared(y_observed, y_predicted, sigma, n_params):
    """Calcula chi-quadrado reduzido"""
    chi_sq = np.sum(((y_observed - y_predicted) / sigma) ** 2)
    degrees_freedom = len(y_observed) - n_params
    return chi_sq / degrees_freedom if degrees_freedom > 0 else np.inf

# Função para ajustar modelos com desvio padrão
def fit_models(t_data, qt_data, sigma_data=None):
    results = {}
    
    # Se não há desvio padrão, usar peso uniforme
    if sigma_data is None:
        sigma_data = np.ones_like(qt_data)
        use_weights = False
    else:
        use_weights = True
    
    # Evitar divisão por zero e valores muito pequenos
    sigma_data = np.where(sigma_data <= 0, np.mean(sigma_data[sigma_data > 0]) if np.any(sigma_data > 0) else 1, sigma_data)
    weights = 1 / (sigma_data ** 2)  # Peso = 1/σ²
    
    # Pseudo-Second Order
    try:
        # Estimativas iniciais
        qe_max = np.max(qt_data)
        k2_init = 0.01
        
        popt_pso, pcov_pso = curve_fit(
            pseudo_second_order, 
            t_data, 
            qt_data, 
            p0=[k2_init, qe_max],
            sigma=sigma_data,
            absolute_sigma=True,
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=5000
        )
        
        qt_pred_pso = pseudo_second_order(t_data, *popt_pso)
        
        # Calcular métricas
        if use_weights:
            r2_pso = weighted_r_squared(qt_data, qt_pred_pso, weights)
            chi2_red_pso = reduced_chi_squared(qt_data, qt_pred_pso, sigma_data, 2)
        else:
            r2_pso = r_squared(qt_data, qt_pred_pso)
            chi2_red_pso = None
        
        # Calcular incertezas dos parâmetros
        param_std_pso = np.sqrt(np.diag(pcov_pso))
        
        results['Pseudo-Second Order'] = {
            'params': {'k2': popt_pso[0], 'qe': popt_pso[1]},
            'param_std': {'k2_std': param_std_pso[0], 'qe_std': param_std_pso[1]},
            'r2': r2_pso,
            'chi2_red': chi2_red_pso,
            'qt_pred': qt_pred_pso,
            'equation': f'qt = ({popt_pso[0]:.4f}±{param_std_pso[0]:.4f}) × ({popt_pso[1]:.4f}±{param_std_pso[1]:.4f})² × t / (1 + ({popt_pso[0]:.4f}±{param_std_pso[0]:.4f}) × ({popt_pso[1]:.4f}±{param_std_pso[1]:.4f}) × t)',
            'n_params': 2
        }
    except Exception as e:
        st.warning(f"Não foi possível ajustar o modelo Pseudo-Second Order: {str(e)}")
        results['Pseudo-Second Order'] = None
    
    # Intraparticle Diffusion
    try:
        popt_ipd, pcov_ipd = curve_fit(
            intraparticle_diffusion, 
            t_data, 
            qt_data,
            sigma=sigma_data,
            absolute_sigma=True,
            maxfev=5000
        )
        
        qt_pred_ipd = intraparticle_diffusion(t_data, *popt_ipd)
        
        # Calcular métricas
        if use_weights:
            r2_ipd = weighted_r_squared(qt_data, qt_pred_ipd, weights)
            chi2_red_ipd = reduced_chi_squared(qt_data, qt_pred_ipd, sigma_data, 2)
        else:
            r2_ipd = r_squared(qt_data, qt_pred_ipd)
            chi2_red_ipd = None
        
        # Calcular incertezas dos parâmetros
        param_std_ipd = np.sqrt(np.diag(pcov_ipd))
        
        results['Intraparticle Diffusion'] = {
            'params': {'kp': popt_ipd[0], 'C': popt_ipd[1]},
            'param_std': {'kp_std': param_std_ipd[0], 'C_std': param_std_ipd[1]},
            'r2': r2_ipd,
            'chi2_red': chi2_red_ipd,
            'qt_pred': qt_pred_ipd,
            'equation': f'qt = ({popt_ipd[0]:.4f}±{param_std_ipd[0]:.4f}) × √t + ({popt_ipd[1]:.4f}±{param_std_ipd[1]:.4f})',
            'n_params': 2
        }
    except Exception as e:
        st.warning(f"Não foi possível ajustar o modelo Intraparticle Diffusion: {str(e)}")
        results['Intraparticle Diffusion'] = None
    
    # Elovich
    try:
        # Filtrar valores de t > 0 para evitar log(0)
        valid_indices = t_data > 0
        t_valid = t_data[valid_indices]
        qt_valid = qt_data[valid_indices]
        sigma_valid = sigma_data[valid_indices]
        
        if len(t_valid) > 2:  # Precisamos de pelo menos 3 pontos
            # Estimativas iniciais
            alpha_init = 1.0
            beta_init = 0.1
            
            popt_elo, pcov_elo = curve_fit(
                elovich, 
                t_valid, 
                qt_valid,
                p0=[alpha_init, beta_init],
                sigma=sigma_valid,
                absolute_sigma=True,
                bounds=([0.001, 0.001], [np.inf, np.inf]),
                maxfev=5000
            )
            
            qt_pred_elo = elovich(t_data, *popt_elo)
            
            # Calcular métricas
            if use_weights:
                weights_valid = 1 / (sigma_valid ** 2)
                qt_pred_valid = elovich(t_valid, *popt_elo)
                r2_elo = weighted_r_squared(qt_valid, qt_pred_valid, weights_valid)
                chi2_red_elo = reduced_chi_squared(qt_valid, qt_pred_valid, sigma_valid, 2)
            else:
                r2_elo = r_squared(qt_data, qt_pred_elo)
                chi2_red_elo = None
            
            # Calcular incertezas dos parâmetros
            param_std_elo = np.sqrt(np.diag(pcov_elo))
            
            results['Elovich'] = {
                'params': {'alpha': popt_elo[0], 'beta': popt_elo[1]},
                'param_std': {'alpha_std': param_std_elo[0], 'beta_std': param_std_elo[1]},
                'r2': r2_elo,
                'chi2_red': chi2_red_elo,
                'qt_pred': qt_pred_elo,
                'equation': f'qt = ({1/popt_elo[1]:.4f}±{param_std_elo[1]/(popt_elo[1]**2):.4f}) × ln(({popt_elo[0]:.4f}±{param_std_elo[0]:.4f}) × ({popt_elo[1]:.4f}±{param_std_elo[1]:.4f})) + ({1/popt_elo[1]:.4f}±{param_std_elo[1]/(popt_elo[1]**2):.4f}) × ln(t)',
                'n_params': 2
            }
        else:
            st.warning("Dados insuficientes para ajustar o modelo Elovich (necessário t > 0)")
            results['Elovich'] = None
    except Exception as e:
        st.warning(f"Não foi possível ajustar o modelo Elovich: {str(e)}")
        results['Elovich'] = None
    
    return results, use_weights

# Interface do usuário
st.sidebar.header("Upload dos Dados")

uploaded_file = st.sidebar.file_uploader(
    "Carregue seu arquivo Excel",
    type=['xlsx', 'xls'],
    help="O arquivo deve conter três colunas: tempo, qe e desvio padrão de qe"
)

if uploaded_file is not None:
    try:
        # Ler arquivo Excel
        df = pd.read_excel(uploaded_file)
        
        st.sidebar.success("Arquivo carregado com sucesso!")
        
        # Mostrar prévia dos dados
        st.subheader("Prévia dos Dados")
        st.dataframe(df.head())
        
        # Seleção de colunas
        st.sidebar.subheader("Seleção de Colunas")
        
        time_col = st.sidebar.selectbox(
            "Selecione a coluna de tempo:",
            df.columns,
            index=0
        )
        
        qt_col = st.sidebar.selectbox(
            "Selecione a coluna de qe:",
            df.columns,
            index=1 if len(df.columns) > 1 else 0
        )
        
        sigma_col = st.sidebar.selectbox(
            "Selecione a coluna de desvio padrão de qe:",
            ['Nenhuma (sem peso)'] + list(df.columns),
            index=2 if len(df.columns) > 2 else 0
        )
        
        # Mostrar informações sobre os dados carregados
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Número de pontos", len(df))
        with col2:
            st.metric("Colunas disponíveis", len(df.columns))
        with col3:
            has_errors = sigma_col != 'Nenhuma (sem peso)'
            st.metric("Desvios padrão", "Sim" if has_errors else "Não")
        
        if sigma_col != 'Nenhuma (sem peso)':
            st.success("✅ Desvios padrão detectados - Será realizado ajuste ponderado")
        else:
            st.info("ℹ️ Sem desvios padrão - Será realizado ajuste com pesos uniformes")
        
        # Opção para mostrar barras de erro
        show_error_bars = st.sidebar.checkbox("Mostrar barras de erro nos gráficos", value=True)
        
        # Botão para limpar cache
        if st.sidebar.button("🧹 Limpar Cache/Resultados", help="Limpa todos os resultados anteriores"):
            for key in list(st.session_state.keys()):
                if key in ['results', 't_data', 'qt_data', 'sigma_data', 'time_col', 'qt_col', 'sigma_col', 'use_weights', 'show_error_bars', 'n_curve_points']:
                    del st.session_state[key]
            st.success("Cache limpo! Carregue seus dados novamente.")
            st.rerun()
        
        # Configurações de exportação
        st.sidebar.subheader("Configurações de Exportação")
        n_curve_points = st.sidebar.slider(
            "Número de pontos nas curvas suaves:",
            min_value=100,
            max_value=1000,
            value=300,
            step=50,
            help="Mais pontos = curvas mais suaves, mas arquivo maior"
        )
        
        if st.sidebar.button("Ajustar Modelos"):
            # LIMPAR DADOS ANTIGOS DO SESSION STATE
            for key in ['results', 't_data', 'qt_data', 'sigma_data', 'time_col', 'qt_col', 'sigma_col', 'use_weights', 'show_error_bars', 'n_curve_points']:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Preparar dados
            t_data = df[time_col].values
            qt_data = df[qt_col].values
            
            # Preparar dados de desvio padrão
            if sigma_col != 'Nenhuma (sem peso)':
                sigma_data = df[sigma_col].values
            else:
                sigma_data = None
            
            # Remover valores NaN
            if sigma_data is not None:
                valid_mask = ~(np.isnan(t_data) | np.isnan(qt_data) | np.isnan(sigma_data))
                sigma_data = sigma_data[valid_mask]
            else:
                valid_mask = ~(np.isnan(t_data) | np.isnan(qt_data))
            
            t_data = t_data[valid_mask]
            qt_data = qt_data[valid_mask]
            
            if len(t_data) < 3:
                st.error("Dados insuficientes para ajuste (mínimo 3 pontos)")
            else:
                # Mostrar informações dos dados carregados para debug
                st.info(f"📊 **Dados carregados:** {len(t_data)} pontos | Tempo: {t_data.min():.2f} - {t_data.max():.2f} | qt: {qt_data.min():.2f} - {qt_data.max():.2f}")
                
                # Ajustar modelos
                peso_text = "com pesos baseados no desvio padrão" if sigma_col != 'Nenhuma (sem peso)' else "com pesos uniformes"
                with st.spinner(f"Ajustando modelos {peso_text}..."):
                    results, use_weights = fit_models(t_data, qt_data, sigma_data)
                
                # Armazenar resultados no session state com validação
                st.session_state.results = results
                st.session_state.t_data = t_data.copy()  # Fazer cópia para evitar referências
                st.session_state.qt_data = qt_data.copy()
                st.session_state.sigma_data = sigma_data.copy() if sigma_data is not None else None
                st.session_state.time_col = time_col
                st.session_state.qt_col = qt_col
                st.session_state.sigma_col = sigma_col
                st.session_state.use_weights = use_weights
                st.session_state.show_error_bars = show_error_bars
                st.session_state.n_curve_points = n_curve_points
                
                # Debug: Mostrar hash dos dados para confirmar
                data_hash = hash(tuple(t_data)) + hash(tuple(qt_data))
                st.session_state.data_hash = data_hash
                st.info(f"📊 **Dados processados com sucesso!** Hash: {data_hash}")
                
                # Mostrar resultado do ajuste
                successful_models = [name for name, data in results.items() if data is not None]
                if successful_models:
                    st.success(f"✅ Ajuste concluído! Modelos ajustados: {', '.join(successful_models)}")
                else:
                    st.error("❌ Nenhum modelo pôde ser ajustado aos dados")
    
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {str(e)}")

# Mostrar resultados se existirem
if 'results' in st.session_state:
    # Validar integridade dos dados
    required_keys = ['results', 't_data', 'qt_data', 'use_weights', 'show_error_bars']
    if not all(key in st.session_state for key in required_keys):
        st.error("❌ Dados inconsistentes detectados. Clique em 'Limpar Cache' e tente novamente.")
        st.stop()
    
    results = st.session_state.results
    t_data = st.session_state.t_data
    qt_data = st.session_state.qt_data
    sigma_data = st.session_state.sigma_data
    use_weights = st.session_state.use_weights
    show_error_bars = st.session_state.show_error_bars
    n_curve_points = st.session_state.get('n_curve_points', 300)
    
    # Mostrar hash para debug
    if 'data_hash' in st.session_state:
        current_hash = hash(tuple(t_data)) + hash(tuple(qt_data))
        if current_hash != st.session_state.data_hash:
            st.error("⚠️ Dados foram alterados! Recalcule os modelos.")
            st.stop()
        else:
            st.success(f"✅ Dados íntegros (Hash: {current_hash})")
    
    # Criar abas para cada modelo
    tab1, tab2, tab3, tab4 = st.tabs(["Resumo", "Pseudo-Second Order", "Intraparticle Diffusion", "Elovich"])
    
    with tab1:
        st.subheader("Resumo dos Ajustes")
        
        if use_weights:
            st.info("📊 **Ajuste ponderado realizado** - Os desvios padrão foram considerados como pesos no ajuste")
        else:
            st.info("📊 **Ajuste sem pesos** - Todos os pontos têm o mesmo peso")
        
        summary_data = []
        for model_name, model_data in results.items():
            if model_data is not None:
                chi2_text = f"{model_data['chi2_red']:.4f}" if model_data['chi2_red'] is not None else "N/A"
                summary_data.append({
                    'Modelo': model_name,
                    'R²': f"{model_data['r2']:.4f}",
                    'χ² reduzido': chi2_text,
                    'Equação': model_data['equation']
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Gráfico comparativo
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Criar pontos suaves para as curvas
            t_smooth = np.linspace(np.min(t_data), np.max(t_data), n_curve_points)
            
            # Dados experimentais com barras de erro
            if sigma_data is not None and show_error_bars:
                ax.errorbar(t_data, qt_data, yerr=sigma_data, fmt='o', 
                           color='black', capsize=3, capthick=1, 
                           label='Dados Experimentais', zorder=5)
            else:
                ax.scatter(t_data, qt_data, color='black', s=50, 
                          label='Dados Experimentais', zorder=5)
            
            # Modelos ajustados com curvas suaves
            colors = ['red', 'blue', 'green']
            for i, (model_name, model_data) in enumerate(results.items()):
                if model_data is not None:
                    # Calcular predições suaves baseadas nos parâmetros ajustados
                    t_curve, qt_curve = generate_smooth_curve(model_name, model_data['params'], t_data, n_curve_points)
                    if t_curve is not None and qt_curve is not None:
                        chi2_text = f", χ² = {model_data['chi2_red']:.3f}" if model_data['chi2_red'] is not None else ""
                        ax.plot(t_curve, qt_curve, 
                               color=colors[i % len(colors)], 
                               linewidth=2, 
                               label=f"{model_name} (R² = {model_data['r2']:.4f}{chi2_text})")
            
            ax.set_xlabel('Tempo')
            ax.set_ylabel('qt')
            ax.set_title('Comparação dos Modelos Cinéticos' + (' (com pesos)' if use_weights else ''))
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        else:
            st.error("Nenhum modelo foi ajustado com sucesso")
    
    # Abas individuais para cada modelo
    model_tabs = [tab2, tab3, tab4]
    model_names = ["Pseudo-Second Order", "Intraparticle Diffusion", "Elovich"]
    
    for tab, model_name in zip(model_tabs, model_names):
        with tab:
            model_data = results.get(model_name)
            
            if model_data is not None:
                st.subheader(f"Modelo {model_name}")
                
                # Parâmetros com incertezas
                st.write("**Parâmetros:**")
                for param, value in model_data['params'].items():
                    param_std_key = f"{param}_std"
                    if param_std_key in model_data['param_std']:
                        uncertainty = model_data['param_std'][param_std_key]
                        st.write(f"- {param}: {value:.6f} ± {uncertainty:.6f}")
                    else:
                        st.write(f"- {param}: {value:.6f}")
                
                st.write(f"**R²:** {model_data['r2']:.6f}")
                if model_data['chi2_red'] is not None:
                    st.write(f"**χ² reduzido:** {model_data['chi2_red']:.6f}")
                st.write(f"**Equação:** {model_data['equation']}")
                
                # Gráfico individual
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Dados experimentais com barras de erro
                if sigma_data is not None and show_error_bars:
                    ax.errorbar(t_data, qt_data, yerr=sigma_data, fmt='o', 
                               color='black', capsize=3, capthick=1, 
                               label='Dados Experimentais')
                else:
                    ax.scatter(t_data, qt_data, color='black', s=50, 
                              label='Dados Experimentais')
                
                # Calcular predições suaves para o modelo
                t_curve, qt_curve = generate_smooth_curve(model_name, model_data['params'], t_data, n_curve_points)
                if t_curve is not None and qt_curve is not None:
                    chi2_text = f", χ² = {model_data['chi2_red']:.3f}" if model_data['chi2_red'] is not None else ""
                    ax.plot(t_curve, qt_curve, color='red', linewidth=2, 
                           label=f'{model_name} (R² = {model_data["r2"]:.4f}{chi2_text})')
                
                ax.set_xlabel('Tempo')
                ax.set_ylabel('qt')
                ax.set_title(f'Ajuste - {model_name}' + (' (com pesos)' if use_weights else ''))
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Gráfico de resíduos
                residuals = qt_data - model_data['qt_pred']
                
                fig_res, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # Resíduos vs tempo
                ax1.scatter(t_data, residuals, color='blue', alpha=0.6)
                ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8)
                ax1.set_xlabel('Tempo')
                ax1.set_ylabel('Resíduos')
                ax1.set_title('Resíduos vs Tempo')
                ax1.grid(True, alpha=0.3)
                
                # Resíduos vs valores preditos
                ax2.scatter(model_data['qt_pred'], residuals, color='green', alpha=0.6)
                ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8)
                ax2.set_xlabel('Valores Preditos')
                ax2.set_ylabel('Resíduos')
                ax2.set_title('Resíduos vs Valores Preditos')
                ax2.grid(True, alpha=0.3)
                
                st.pyplot(fig_res)
                
            else:
                st.error(f"Modelo {model_name} não pôde ser ajustado")
    
    # Exportar resultados
    st.subheader("Exportar Resultados")
    
    if st.button("Gerar Arquivo de Exportação"):
        # Criar DataFrame com todos os resultados
        export_data = {
            st.session_state.time_col: t_data,
            st.session_state.qt_col: qt_data
        }
        
        if sigma_data is not None:
            export_data[f'desvio_padrao_{st.session_state.qt_col}'] = sigma_data
        
        # Adicionar predições nos pontos experimentais
        for model_name, model_data in results.items():
            if model_data is not None:
                export_data[f'{model_name}_pred_experimental'] = model_data['qt_pred']
                export_data[f'{model_name}_residuals'] = qt_data - model_data['qt_pred']
        
        export_df = pd.DataFrame(export_data)
        
        # Gerar curvas suaves para exportação
        st.info(f"📊 Gerando curvas com {n_curve_points} pontos para exportação...")
        
        curves_data = {}
        t_smooth = np.linspace(np.min(t_data), np.max(t_data), n_curve_points)
        curves_data['tempo_curva'] = t_smooth
        
        curves_info = []
        for model_name, model_data in results.items():
            if model_data is not None:
                t_curve, qt_curve = generate_smooth_curve(model_name, model_data['params'], t_data, n_curve_points)
                if t_curve is not None and qt_curve is not None:
                    curves_data[f'{model_name}_curva'] = qt_curve
                    curves_info.append(f"{model_name}: {len(qt_curve)} pontos")
        
        if curves_info:
            st.success(f"✅ Curvas geradas: {', '.join(curves_info)}")
        else:
            st.warning("⚠️ Nenhuma curva foi gerada")
        
        curves_df = pd.DataFrame(curves_data)
        
        # Criar arquivo Excel em memória
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Aba com dados experimentais e predições nos pontos experimentais
            export_df.to_excel(writer, sheet_name='Dados_Experimentais', index=False)
            
            # Aba com curvas suaves para gráficos
            curves_df.to_excel(writer, sheet_name='Curvas_Ajustadas', index=False)
            
            # Aba com parâmetros e incertezas detalhada
            params_data = []
            for model_name, model_data in results.items():
                if model_data is not None:
                    # Adicionar uma linha de cabeçalho para cada modelo
                    params_data.append({
                        'Modelo': f'=== {model_name} ===',
                        'Parâmetro': '',
                        'Valor': '',
                        'Incerteza': '',
                        'R²': '',
                        'Chi² Reduzido': ''
                    })
                    
                    # Parâmetros com incertezas
                    for param, value in model_data['params'].items():
                        param_std_key = f"{param}_std"
                        uncertainty = model_data['param_std'].get(param_std_key, 0)
                        params_data.append({
                            'Modelo': model_name,
                            'Parâmetro': param,
                            'Valor': value,
                            'Incerteza': uncertainty,
                            'R²': '',
                            'Chi² Reduzido': ''
                        })
                    
                    # Métricas na mesma linha
                    chi2_value = model_data['chi2_red'] if model_data['chi2_red'] is not None else 'N/A'
                    params_data.append({
                        'Modelo': model_name,
                        'Parâmetro': 'Métricas',
                        'Valor': '',
                        'Incerteza': '',
                        'R²': model_data['r2'],
                        'Chi² Reduzido': chi2_value
                    })
                    
                    # Equação
                    params_data.append({
                        'Modelo': model_name,
                        'Parâmetro': 'Equação',
                        'Valor': model_data['equation'],
                        'Incerteza': '',
                        'R²': '',
                        'Chi² Reduzido': ''
                    })
                    
                    # Linha em branco para separar modelos
                    params_data.append({
                        'Modelo': '',
                        'Parâmetro': '',
                        'Valor': '',
                        'Incerteza': '',
                        'R²': '',
                        'Chi² Reduzido': ''
                    })
            
            params_df = pd.DataFrame(params_data)
            params_df.to_excel(writer, sheet_name='Parametros_e_Incertezas', index=False)
            
            # Aba adicional com resumo estatístico
            summary_data = []
            if use_weights:
                summary_data.append({
                    'Informação': 'Tipo de Ajuste',
                    'Valor': 'Ajuste Ponderado (com pesos baseados no desvio padrão)'
                })
                summary_data.append({
                    'Informação': 'Função Peso',
                    'Valor': 'w = 1/σ²'
                })
            else:
                summary_data.append({
                    'Informação': 'Tipo de Ajuste',
                    'Valor': 'Ajuste Sem Pesos (peso uniforme)'
                })
            
            summary_data.append({
                'Informação': 'Número de Pontos Experimentais',
                'Valor': len(t_data)
            })
            
            summary_data.append({
                'Informação': 'Número de Pontos nas Curvas',
                'Valor': n_curve_points
            })
            
            summary_data.append({
                'Informação': 'Melhor Modelo (R²)',
                'Valor': max([(name, data['r2']) for name, data in results.items() if data is not None], 
                           key=lambda x: x[1])[0] if any(data is not None for data in results.values()) else 'N/A'
            })
            
            if use_weights:
                best_chi2_model = min([(name, data['chi2_red']) for name, data in results.items() 
                                     if data is not None and data['chi2_red'] is not None], 
                                    key=lambda x: x[1], default=('N/A', 'N/A'))
                summary_data.append({
                    'Informação': 'Melhor Modelo (χ²)',
                    'Valor': best_chi2_model[0]
                })
            
            # Informações sobre as abas do arquivo
            summary_data.append({
                'Informação': '--- ESTRUTURA DO ARQUIVO ---',
                'Valor': ''
            })
            summary_data.append({
                'Informação': 'Aba: Dados_Experimentais',
                'Valor': 'Dados originais + predições nos pontos experimentais + resíduos'
            })
            summary_data.append({
                'Informação': 'Aba: Curvas_Ajustadas',
                'Valor': f'Curvas suaves com {n_curve_points} pontos para gráficos'
            })
            summary_data.append({
                'Informação': 'Aba: Parametros_e_Incertezas',
                'Valor': 'Parâmetros ajustados com suas incertezas e métricas'
            })
            summary_data.append({
                'Informação': 'Aba: Resumo_Ajuste',
                'Valor': 'Esta aba - informações gerais sobre o ajuste'
            })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Resumo_Ajuste', index=False)
        
        output.seek(0)
        
        # Botão de download
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        weighted_text = "_ponderado" if use_weights else "_sem_peso"
        st.download_button(
            label="📥 Baixar Resultados (Excel)",
            data=output.getvalue(),
            file_name=f"ajuste_cinetico{weighted_text}_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Mostrar informações sobre o arquivo exportado
        st.success("✅ Arquivo gerado com sucesso!")
        st.info("""
        **📋 Estrutura do arquivo Excel:**
        
        📸 **Dados_Experimentais:** Seus dados originais + predições nos pontos experimentais + resíduos
        
        📸 **Curvas_Ajustadas:** Curvas suaves para criar gráficos elegantes (tempo_curva + curvas de cada modelo)
        
        📸 **Parametros_e_Incertezas:** Parâmetros ajustados com incertezas, R², χ² e equações
        
        📸 **Resumo_Ajuste:** Informações gerais sobre o tipo de ajuste e configurações
        """)

else:
    st.info("👆 Carregue um arquivo Excel na barra lateral para começar a análise")
    
    # Informações sobre os modelos
    st.subheader("Modelos Cinéticos Disponíveis")
    
    st.markdown("""
    **1. Pseudo-Second Order:**
    - Equação: qt = (k₂ × qₑ² × t) / (1 + k₂ × qₑ × t)
    - Parâmetros: k₂ (constante de velocidade), qₑ (capacidade de equilíbrio)
    
    **2. Intraparticle Diffusion:**
    - Equação: qt = kₚ × √t + C
    - Parâmetros: kₚ (constante de difusão), C (intercepto)
    
    **3. Elovich:**
    - Equação: qt = (1/β) × ln(α × β) + (1/β) × ln(t)
    - Parâmetros: α (velocidade inicial de adsorção), β (constante de dessorção)
    """)
    
    st.subheader("Formato do Arquivo")
    st.markdown("""
    - O arquivo deve estar em formato Excel (.xlsx ou .xls)
    - **Colunas esperadas:** tempo, qe, desvio padrão de qe
    - Os dados devem estar organizados em colunas
    - Valores faltantes serão automaticamente removidos
    - O desvio padrão será usado como peso no ajuste (peso = 1/σ²)
    """)
    
    st.subheader("Ajuste com Pesos")
    st.markdown("""
    **Quando você fornecer os desvios padrão:**
    - Os pontos com menor incerteza terão maior peso no ajuste
    - Será calculado o χ² reduzido para avaliar a qualidade do ajuste
    - As incertezas dos parâmetros serão calculadas
    - Os gráficos mostrarão barras de erro
    - O R² será calculado de forma ponderada
    
    **Métricas de qualidade:**
    - **R² ponderado:** Coeficiente de determinação considerando os pesos
    - **χ² reduzido:** Deve estar próximo de 1 para um bom ajuste
    - **Incertezas dos parâmetros:** Propagação de erros através da matriz de covariância
    """)
    
    st.subheader("Novo! 🎯 Exportação Aprimorada")
    st.markdown("""
    **O arquivo Excel agora inclui:**
    
    📊 **Curvas Ajustadas:** Dados suaves para criar gráficos profissionais
    
    📈 **Duas escalas de dados:**
    - Predições nos pontos experimentais (para análise de resíduos)
    - Curvas suaves interpoladas (para visualização gráfica)
    
    ⚙️ **Controle de resolução:** Configure quantos pontos usar nas curvas (100-1000 pontos, padrão: 300)
    
    📁 **Estrutura organizada:** Cada tipo de informação em abas separadas para facilitar o uso
    """)
